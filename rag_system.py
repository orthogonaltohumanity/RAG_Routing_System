from os import listdir
from os.path import isfile, join, exists
import numpy as np

#Chroma
import chromadb

# LangChain imports
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from tqdm import tqdm
#HuggingFace
from datasets import load_dataset
from langchain_community.document_loaders.hugging_face_dataset import HuggingFaceDatasetLoader

def find_pdf_files(directory_path: str) -> list[str]:
    """Finds all PDF files in the specified directory."""
    pdf_files = []
    try:
        for filename in listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                full_path = join(directory_path, filename)
                if isfile(full_path):
                    pdf_files.append(full_path)
    except OSError as e:
        logging.error(f"Error accessing directory {directory_path}: {e}")
        raise  # Re-raise the exception after logging
    if len(pdf_files)>0:
        return pdf_files
    else:
        print("No PDFs Found In Directory")
        return []

def find_json_files(directory_path: str) -> list[str]:
    """Finds all JSON files in the specified directory."""
    json_files = []
    try:
        for filename in listdir(directory_path):
            if filename.lower().endswith(".json"):
                full_path = join(directory_path, filename)
                if isfile(full_path):
                    json_files.append(full_path)
    except OSError as e:
        logging.error(f"Error accessing directory {directory_path}: {e}")
        raise  # Re-raise the exception after logging
    return json_files





#Transformers Import for HuggingFaceEmbeddings Workaround
#from transformers import AutoModel
class RagSystem:
    def __init__(
            self,
            embed_model: str = "BAAI/llm-embedder",
            llm_model: str = "douglas_v3",
            chunk_size: int = 1200,
            chunk_overlap: int = 300,
            collection:str="default",
            rag_threshhold:float=0.5,
            retriever_template:str="""You are an AI assistant. Your role is to assist the user in finding new information by pulling from a vector database. You will also learn from the information provided.

                Original question: {question}""",
            rag_template:str="""Use the context provided and your own knowledge to answer any questions. Do not give long responses. You will give short response.

                {context}
                
                Question: {question}
                
                If the information cannot be found in the context, say so. You can still answer the question, but just say if its not in the context.
                If you are asked to provide a source then include the source of the information in your answer when possible.
                """
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        self.llm = None
        self.chain = None
        self.collection=collection
        self.rag_threshhold=rag_threshhold
        self.retriever_template=retriever_template
        self.rag_template=rag_template

    def load_all(self,pdfs_dir:list,hugs_dir:list,hug_cache:str,hug_cols:list):
        chunks=[]
        chunks.extend(self.load_pdfs(pdfs_dir))
        chunks.extend(self.load_huggingface_dataset(file_paths=hugs_dir,
                                                     cache_dir=hug_cache,
                                                     columns=hug_cols))
        return chunks
    def load_chroma(self):
        client=chromadb.PersistentClient(path="chroma_db")
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model,model_kwargs={'device':'cuda'},cache_folder="HGEmbeddings")
        self.vector_db=Chroma(client=client,collection_name=self.collection,embedding_function=embeddings)

    def load_huggingface_dataset(self,file_paths:list,cache_dir:str,columns:list):
        all_chunks=[]
        for file_path,column in zip(file_paths,columns):
            try:
                loader=HuggingFaceDatasetLoader(path=file_path,cache_dir=cache_dir,page_content_column=column)
                doc=loader.lazy_load()
                #for d in doc_raw:
                #    doc.append(d.page_content)
                
                splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = splitter.split_documents(documents=doc)
                chunks=filter_complex_metadata(chunks)
                print(f"{file_path} split into {len(chunks)} chunks.")

                all_chunks.extend(chunks)

            except Exception as e:
                raise Exception(f"Error loading JSON {file_path}: {str(e)}")

        if not all_chunks:
            print("No valid JSONSs were loaded.")
            return []

        print(f"Total chunks from JSONSs: {len(all_chunks)}")
        return all_chunks
    

    def load_messages(self,messages:list):
        
        content=messages[:][1]
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        docs=text_splitter.create_documents(content)
        chunks=text_splitter.split_documents(documents=docs)
    
        return chunks

    def compress_vector_db(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model,model_kwargs={'device':'cuda'},cache_folder="HGEmbeddings")
        if not self.vector_db:
            print("Vector DB not initialized.")
            model_kwargs={'trust_remote_code':True}
            try:
                client=chromadb.PersistentClient(path="chroma_db")
            #collection=client.get_or_create_collection(name="multi-pdf-rag")
                self.vector_db=Chroma(client=client,collection_name=self.collection,embedding_function=embeddings)
            #self.vector_db=collection.get()
                print("Vector DB found and loaded from disk.")
            except Exception as e:
                print("No Vector DB found.")
                print(e)
        try:
            self.setup_rag_chain()
            compression=self.query("Summarize all data inside this database.")
            
            client=chromadb.PersistentClient(path="chroma_db")
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            docs=text_splitter.create_documents([compression])
            
            self.vector_db=Chroma.from_documents(client=client,collection_name=self.collection,documents=docs,embedding=embeddings)
        except Exception as e:
            print(e)


    def load_pdfs(self, file_paths: list):
        all_chunks = []

        for file_path in file_paths:
            if not exists(file_path):
                print(f"Warning: PDF file not found - {file_path}, skipping...")
                continue

            try:
                print(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(file_path=file_path)
                doc = loader.lazy_load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = splitter.split_documents(documents=doc)
                print(f"{file_path} split into {len(chunks)} chunks.")

                all_chunks.extend(chunks)

            except Exception as e:
                raise Exception(f"Error loading PDF {file_path}: {str(e)}")

        if not all_chunks:
            print("No valid PDFs were loaded.")
            return []
        print(f"Total chunks from PDFs: {len(all_chunks)}")
        return all_chunks


    def create_vector_db(self, chunks,add_to_existing:bool=False):
        try:
            print("Creating vector database...")
            #model=AutoModel.from_pretrained(pretrained_model_name_or_path=self.embed_model,trust_remote_code=True)
            
            
            embeddings = HuggingFaceEmbeddings(model_name=self.embed_model,model_kwargs={'device':'cuda'},cache_folder="HGEmbeddings")
            print("Embedding Function Initialized")
            client=chromadb.PersistentClient(path="chroma_db")
            if not add_to_existing:
                self.vector_db = Chroma.from_documents(
                    client=client,
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=self.collection,
                    #persist_directory="chroma_db"
                )
            else:
                self.vector_db=Chroma(client=client,collection_name=self.collection,embedding_function=embeddings)
                self.vector_db.add_documents(documents=chunks)
            


            print("Vector DB Saved to Disk")
            print(f"Vector DB created with collection_name: {self.collection}.")
        except Exception as e:
            raise Exception(f"Error creating database: {str(e)}")
    def check_similarity(self,prompt:str):
        if not self.vector_db:
            print("Vector DB not initialized.")
            model_kwargs={'trust_remote_code':True}
            embeddings = HuggingFaceEmbeddings(model_name=self.embed_model,model_kwargs={'device':'cuda'},cache_folder="HGEmbeddings")
            try:
                client=chromadb.PersistentClient(path="chroma_db")
            #collection=client.get_or_create_collection(name="multi-pdf-rag")
                self.vector_db=Chroma(client=client,collection_name=self.collection,embedding_function=embeddings)
            #self.vector_db=collection.get()
                print("Vector DB found and loaded from disk.")
            except Exception as e:
                print("No Vector DB found.")
                print(e)
        results_with_scores=self.vector_db.similarity_search_with_relevance_scores(prompt)
        avg_value=0.0  # Initialize with the first element's second entry
        for tup in results_with_scores:
            avg_value+=tup[1]
        return avg_value/len(results_with_scores)
    def setup_rag_chain(self):
        if not self.vector_db:
            print("Vector DB not initialized.")
            model_kwargs={'trust_remote_code':True}
            embeddings = HuggingFaceEmbeddings(model_name=self.embed_model,model_kwargs={'device':'cuda'},cache_folder="HGEmbeddings")
            try:
                client=chromadb.PersistentClient(path="chroma_db")
            #collection=client.get_or_create_collection(name="multi-pdf-rag")
                self.vector_db=Chroma(client=client,collection_name=self.collection,embedding_function=embeddings)
            #self.vector_db=collection.get()
                print("Vector DB found and loaded from disk.")
            except Exception as e:
                print("No Vector DB found.")
                print(e)
            
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=0.4,
                top_p=0.8,
                keep_alive=0.0
            )

            query_prompt = PromptTemplate(
                input_variables=["question"],
                template=self.retriever_template,
            )

            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 4}),
                llm=self.llm,
                prompt=query_prompt
            )

            template = self.rag_template

            prompt = ChatPromptTemplate.from_template(template=template)

            self.chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )

            print("RAG chain setup complete!")

        except Exception as e:
            raise Exception(f"Error setting up RAG chain: {str(e)}")

    def query(self, question: str):
        if not self.chain:
            raise ValueError("RAG chain not initialized.")

        try:
            print(f"Processing")
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
