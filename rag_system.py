from os import listdir
from os.path import isfile, join, exists

# LangChain imports
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


class PdfRagSystem:
    def __init__(
            self,
            embed_model: str = "mxbai-embed-large",
            llm_model: str = "phi4:14b-q4_K_M",
            chunk_size: int = 1200,
            chunk_overlap: int = 300,
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        self.llm = None
        self.chain = None

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
            raise ValueError("No valid PDFs were loaded.")

        print(f"Total chunks from PDFs: {len(all_chunks)}")
        return all_chunks

    def create_vector_db(self, chunks, collection_name: str = "multi-pdf-rag"):
        try:
            print("Creating vector database...")
            embeddings = OllamaEmbeddings(model=self.embed_model)

            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name
            )
            print(f"Vector DB created with collection_name: {collection_name}.")
        except Exception as e:
            raise Exception(f"Error creating database: {str(e)}")

    def setup_rag_chain(self):
        if not self.vector_db:
            raise ValueError("Vector DB not initialized.")

        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=0.3,
                top_p=0.8
            )

            query_prompt = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
            )

            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 4}),
                llm=self.llm,
                prompt=query_prompt
            )

            template = """Answer the question based ONLY on the following context:
                {context}
                
                Question: {question}
                
                If the information cannot be found in the context, say "I don't have enough information to answer this question."
                Include the source of the information in your answer when possible.
                """

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
            print(f"Processing query: '{question}'")
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
