import numpy as n
from sklearn.metrics import pairwise_distances
from langchain_ollama.chat_models import ChatOllama
from rag_system import RagSystem
import cv2
import ollama
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

class llmNetwork:
    def __init__(self,
                breakdown_model,
                ideas_model,
                think_model,
                verify_model,
                response_model,
                memoryRAG,
                 #emotionRAG,
                vision_model,
                huggingface_sentiment_model,
                huggingface_emotion_model,
                name:str="AI",
                breakdown_prompt:str="You are an AI. Please thoroughly summarize the context and message history so that it can be easily understood and followed. Prioritize more recent information.",
                ideas_prompt:str="You are an AI. Take the summarization and give three potential thinking processes someone could use to respond to the users prompt. Feel free to be creative and generate novel ideas, but be sure to always state your assumptions.",
                think_prompt:str="You are an AI. Please take the context and the users thinking processes and follow them through to come to a conclusion. Keep in mind the assumptions you are working with.",
                verify_prompt:str="You are an AI. Please take the context and verify the reasoning you are given is sound, without circular logic, and based in fact.",
                response_prompt:str="You are an AI. Take the context, your thought process, and the verification of your thought process and use it to respond to the user.",
                memory_query:str="You are an AI pulling from a vector database of past chat messages. Give a brief summary of the conversation.",

                 ):
        self.modules=[]

        self.breakdown_model=breakdown_model
        self.ideas_model=ideas_model
        self.think_model=think_model
        self.verify_model=verify_model
        self.response_model=response_model

        self.name=name
        self.memoryRAG=memoryRAG
        #self.emotionRAG=emotionRAG

        self.breakdown_prompt=breakdown_prompt
        self.ideas_prompt=ideas_prompt
        self.think_prompt=think_prompt
        self.verify_prompt=verify_prompt
        self.response_prompt=response_prompt

        self.memory_query=memory_query
        self.vision_model=vision_model
        self.huggingface_sentiment_model=huggingface_sentiment_model
        self.huggingface_emotion_model=huggingface_emotion_model
            ##################################################################
    def rag_query(self,prompt:str):
        max_score=0.0
        chosen_module=self.modules[0]
        for module in self.modules:
            score=module.check_similarity(prompt)
            #print("Module "+module.collection+" Score "+str(score))
            if score > max_score:
                max_score=score
                chosen_module=module
        print("Module: "+chosen_module.collection)
        print("Model: "+chosen_module.llm_model)
        chosen_module.setup_rag_chain()
        return chosen_module.query(prompt)
    def chat(self,messages:list):
        prompt=messages[-1][1]

        
        image_description=""
        try:     
            print("Looking around")
            image_description=self.get_image_and_process(prompt) 

        except Exception as e:
            print(e)



        sentiment=""
        emotion=""
        intake_sentiment=""
        intake_emotion=""
        memories=""
        topic="" 
        try:
            for module in self.modules:
                if(module.check_similarity(prompt+","+image_description)>module.rag_threshhold):
                    print("Collecting Relevant Information from "+module.collection)
                    module.setup_rag_chain()
                    
                    topic+=module.query("Find information relevant to the following prompt and image description\n"+prompt+","+image_description)
 
        except Exception as e:
            print(e)

        try:

            print("Remembering Past Conversations")
            self.memoryRAG.setup_rag_chain()
            memories= self.memoryRAG.query(self.memory_query+" Find queries relevant to this prompt: "+prompt)
        except Exception as e:
            print(e)
        try:
            print("Analyzing Prompt Emotional Sentiment")
            E_tokenizer=AutoTokenizer.from_pretrained(self.huggingface_sentiment_model, cache_dir="HGSentiment")
            E_model=AutoModelForSequenceClassification.from_pretrained(self.huggingface_sentiment_model,cache_dir="HGSentiment")
            inputs=E_tokenizer(prompt,return_tensors="pt")
            with torch.no_grad():
                logits=E_model(**inputs).logits

            predicted_class_id=logits.argmax().item()
            sentiment=E_model.config.id2label[predicted_class_id]
        except Exception as e:
            print(e)
        try:
            print("Classifying Prompt Emotions")

            E_tokenizer=AutoTokenizer.from_pretrained(self.huggingface_emotion_model, cache_dir="HGSentiment")
            E_model=AutoModelForSequenceClassification.from_pretrained(self.huggingface_emotion_model,cache_dir="HGSentiment")
            inputs=E_tokenizer(prompt,return_tensors="pt")
            with torch.no_grad():
                logits=E_model(**inputs).logits
            #predicted_class_id=logits.argmax().item()

            predicted_class_id=torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
            emotion=""
            for t in predicted_class_id:
                emotion=emotion+E_model.config.id2label[t.item()]+","
        except Exception as e:
            print(e)
       
        intake=topic+"\n"+memories+"\n"+image_description
        intake=intake[0:514]


        try:   
            print("Analyzing Intake Emotional Sentiment")
            E_tokenizer=AutoTokenizer.from_pretrained(self.huggingface_sentiment_model, cache_dir="HGSentiment")
            E_model=AutoModelForSequenceClassification.from_pretrained(self.huggingface_sentiment_model,cache_dir="HGSentiment")
            inputs=E_tokenizer(intake,return_tensors="pt")
            with torch.no_grad():
                logits=E_model(**inputs).logits
            predicted_class_id=logits.argmax().item()
            intake_sentiment=E_model.config.id2label[predicted_class_id]


            #intake_sentiment=E_model.config.id2label[predicted_class_id]
        except Exception as e:
            print(e)
        try:
            print("Classifying Intake Prompt Emotions")

            E_tokenizer=AutoTokenizer.from_pretrained(self.huggingface_emotion_model, cache_dir="HGSentiment")
            E_model=AutoModelForSequenceClassification.from_pretrained(self.huggingface_emotion_model,cache_dir="HGSentiment")
            inputs=E_tokenizer(intake,return_tensors="pt")

            with torch.no_grad():
                logits=E_model(**inputs).logits
            predicted_class_id=torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

            intake_emotion=""
            for t in predicted_class_id:
                intake_emotion=intake_emotion+E_model.config.id2label[t.item()]+","


        except Exception as e:
            print(e)

        print("Breaking Down Context")
        system_input=[("system","Your name is "+self.name+
        "\n"+self.breakdown_prompt+
        "You can see the following:\n"+image_description+
        "\n The sentiments of the  users prompt is: "+sentiment+
        "\n The most prevalent emotions in the users prompt is: "+emotion+
        "\n The sentiments of your current experiences is: "+intake_sentiment+
        "\n The emotions of your current experiences is: "+intake_emotion+
        "\nSummary of old conversations you had: "+memories+
        "\nSome information from a database relevant to the conversation : "+topic)]+messages
        
        breakdown_llm=ChatOllama(model=self.breakdown_model,
                                 temperature=0.3,
                                 top_p=0.8,
                                 keep_alive=0.0,
                                 num_ctx=10000
                                 )

        breakdown=breakdown_llm.invoke(system_input).content

        print("Generating Ideas")
        ideas_llm=ChatOllama(model=self.ideas_model,
                                 temperature=0.7777777,
                                 top_p=0.8,
                                 keep_alive=0.0,
                                 num_ctx=10000
                                 )

        system_input=[("system",self.ideas_prompt),("user",breakdown)]
        ideas=ideas_llm.invoke(system_input).content


        print("Thinking")
        think_llm=ChatOllama(model=self.think_model,
                                 temperature=0.5,
                                 top_p=0.8,
                                 keep_alive=0.0,
                                 num_ctx=10000
                            )
        

        system_input=[("system",self.think_prompt+breakdown),("user",ideas)]
        thought=think_llm.invoke(system_input).content
        
        know_check=""

        print("Reassessing Knowledge Base")
        try:
            for module in self.modules:
                if(module.check_similarity(thought)>module.rag_threshhold):
                    print("Collecting Relevant Information from "+module.collection)
                    module.setup_rag_chain()
                    
                    know_check+=module.query("Find information relevant to the following prompt and image description\n"+prompt+","+image_description)
 
        except Exception as e:
            print(e)



        print("Performing Metacognition")
        verify_llm=ChatOllama(model=self.verify_model,
                                 temperature=0.1,
                                 top_p=0.8,
                                 keep_alive=0.0,
                                 num_ctx=10000)

 

        system_input=[("system",self.verify_prompt+breakdown+know_check),("user",thought)]
        verify=verify_llm.invoke(system_input).content

        print("Generating Response")
        messages=[("system","Your name is "+self.name+
        "\n"+self.response_prompt+
        "You can see the following:\n"+image_description+
        "\n The sentiments of the  users prompt is: "+sentiment+
        "\n The most prevalent emotions in the users prompt is: "+emotion+
        "\n The sentiments of your current experiences is: "+intake_sentiment+
        "\n The emotions of your current experiences is: "+intake_emotion+
        "\nSummary of old conversations you had: "+memories+
        "\nSome information from a database relevant to the conversation : "+topic+
        "\nYour thought process: "+thought+
        "\nYour verification of your thought process"+verify)]+messages

        response_llm=ChatOllama(model=self.response_model,
                                temperature=0.3,
                                top_p=0.8,
                                keep_alive=0.0,
                                num_ctx=10000)

        response= response_llm.invoke(messages).content



        return response

    def query(self,prompt:str):
        max_score=0.0
        chosen_module=self.modules[0]
        for module in self.modules:
            score=module.check_similarity(prompt)
            #print("Module "+module.collection+" Score "+str(score))
            if score > max_score:
                max_score=score
                chosen_module=module
        print("Module: "+chosen_module.collection)
        print("Model: "+chosen_module.llm_model)
        
        if max_score<chosen_module.rag_threshhold:
            chosen_module.llm = ChatOllama(model=chosen_module.llm_model,
                                    temperature=0.1,
                                    top_p=0.8,
                                    keep_alive=0.0,
                                    num_ctx=5000)
           
            messages=[
                    (
                        "system",
                        "You are a helpful AI assistant who gives short but complete summarizations"
                    ),
                    (
                        "human",
                        prompt
                    )
                    ]
            return chosen_module.llm.invoke(messages).content
        else:
            chosen_module.setup_rag_chain()
            return chosen_module.query(prompt)
    def add_rag_system(self,rag_sys_list:list):
        for rag_sys in rag_sys_list:
            self.modules.append(rag_sys)
    def get_image_and_process(self,prompt):
        vid=cv2.VideoCapture(0)
        ret,frame=vid.read()
        #frame=cv2.resize(frame,(120,120))
        cv2.imwrite("Images/vision.jpg",frame)
        try: 
            image_description=ollama.chat(self.vision_model,messages=[{'role':'system','content':'You are a vision AI. You will describe what you see in detail. Repeat back any writing you have detected. Finally, look for information relavant to this prompt: '+prompt,'images':['Images/vision.jpg']}],keep_alive=0.0)
            #print(image_description.message.content)
            return image_description.message.content
        except Exception as e:
            print(e)
            return "Pure Black"
