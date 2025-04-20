from rag_system import RagSystem
from multi import llmNetwork
from rag_system import find_pdf_files


import string
import os,sys
from contextlib import contextmanager
import pyttsx3
import speech_recognition as sr
import pyaudio
import torch
from TTS.api import TTS
import wave
import sounddevice

def write_memories_to_disk(memoryRAG,messages):
    try:
        chunks=memoryRAG.load_messages(messages)
        memoryRAG.create_vector_db(chunks,add_to_existing=True)
        
    except Exception as e:
        print(e)

def compress_memories(memoryRAG,messages):
    try:
        write_memories_to_disk(memoryRAG,messages)
        memoryRAG.compress_vector_db()
        print("Memories Compressed!")
    except Exception as e:
        print(e)
def suppress_stderr():
    pass

def Speech_Engine():
    engine=pyttsx3.init()
    return engine

def Speak(text):
    device="cuda" if torch.cuda.is_available() else "cpu"
    #device="cpu"
    if device=="cpu":print("Offloading Voice Model to CPU")
    text=text.replace('*','')
    #tts=TTS("tts_models/en/jenny/jenny").to(device)
    tts=TTS("tts_models/en/ljspeech/neural_hmm").to(device)
    filename='voice_cache/tts.wav'
    tts.tts_to_file(text=text,file_path=filename)
    wf=wave.open(filename)
    p=pyaudio.PyAudio()
    stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)
    data=wf.readframes(1024)
    print(text)
    while data:
        stream.write(data)
        data=wf.readframes(1024)
    wf.close()
    stream.close()
    p.terminate()

def Listen():
    engine=Speech_Engine()
    recog=sr.Recognizer()
    with sr.Microphone() as source:
        recog.adjust_for_ambient_noise(source,duration=0.2)
        sound=recog.listen(source,timeout=5000,phrase_time_limit=5000)
        return recog.recognize_whisper(sound,language="english")

memoryRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="memories",
            rag_threshhold=0.99
        )
        
#memory_cache_path="Docs/Juniper_memories"
'''
print("Loading From Emotion Database")
emotionRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="emotions",
            rag_threshhold=0.99
        )
        
emotion_cache_path="Docs/emotions"
emotion_pdf_paths=[]
emotion_hug_paths=["dair-ai/emotion","dair-ai/emotion"]
emotion_hug_cols=["text","label"]
'''
'''
chunks=emotionRAG.load_all(pdfs_dir=emotion_pdf_paths,
                             hugs_dir=emotion_hug_paths,
                             hug_cache=emotion_cache_path,
                             hug_cols=emotion_hug_cols)
emotionRAG.create_vector_db(chunks)
'''
       
mathRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "deepseek-r1:1.5b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="math",
            rag_threshhold=0.8
        ) 
philRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="phil",    
            rag_threshhold=0.8
        ) 
italyRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="italy",    
            rag_threshhold=0.8
        )
       
us_histRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="us_hist",
            rag_threshhold=0.8
        )

fictionRAG=RagSystem(
            embed_model= "BAAI/llm-embedder",
            llm_model = "gemma3:1b",
            chunk_size = 1200,
            chunk_overlap = 300,
            collection ="fiction",
            rag_threshhold=0.8
        )


math_cache_path="Docs/math"
phil_cache_path="Docs/phil"
italy_cache_path="Docs/italy"
us_hist_cache_path="Docs/us_history"
fiction_cache_path="Docs/fiction"


math_pdf_paths=find_pdf_files(math_cache_path)
phil_pdf_paths=find_pdf_files(phil_cache_path)
italy_pdf_paths=find_pdf_files(italy_cache_path)
us_hist_pdf_paths=find_pdf_files(us_hist_cache_path)
fiction_pdf_paths=find_pdf_files(fiction_cache_path)

math_hug_paths=[]#["qwedsacf/competition_math"]
phil_hug_paths=[]#["Heigke/stanford-enigma-philosophy-chat"]
italy_hug_paths=[]
us_hist_hug_paths=[]
fiction_hug_paths=[]


math_hug_cols=["problem"]
phil_hug_cols=["output"]
italy_hug_cols=[]
us_hist_hug_cols=[]
fiction_hug_cols=[]

load_from_files=False
if load_from_files:
    #===============================================
    print("Loading First LLM")
    chunks=us_histRAG.load_all(pdfs_dir=us_hist_pdf_paths,
                             hugs_dir=us_hist_hug_paths,
                             hug_cache=us_hist_cache_path,
                             hug_cols=us_hist_hug_cols)
    us_histRAG.create_vector_db(chunks)

   #===============================================
    print("Loading Second LLM")
    chunks=mathRAG.load_all(pdfs_dir=math_pdf_paths,
                             hugs_dir=math_hug_paths,
                             hug_cache=math_cache_path,
                             hug_cols=math_hug_cols)
    mathRAG.create_vector_db(chunks)
    #===============================================
    print("Loading Third LLM")
    chunks=philRAG.load_all(pdfs_dir=phil_pdf_paths,
                             hugs_dir=phil_hug_paths,
                             hug_cache=phil_cache_path,
                             hug_cols=phil_hug_cols)
    philRAG.create_vector_db(chunks)
    print("Loading Fourth LLM")
    chunks=italyRAG.load_all(pdfs_dir=italy_pdf_paths,
                             hugs_dir=italy_hug_paths,
                             hug_cache=italy_cache_path,
                             hug_cols=italy_hug_cols)
    italyRAG.create_vector_db(chunks)
    print("Loading Fifth LLM")
    chunks=fictionRAG.load_all(pdfs_dir=fiction_pdf_paths,
                             hugs_dir=fiction_hug_paths,
                             hug_cache=fiction_cache_path,
                             hug_cols=fiction_hug_cols)
    fictionRAG.create_vector_db(chunks)


print("======================VcB Loaded=====================")
Brain=llmNetwork(
                 breakdown_model="qwen2.5:3b",
                 ideas_model="granite3.2:2b",
                 think_model="granite3.2:2b",
                 verify_model="granite3.2:2b",
                 response_model="gemma3:4b",
                 memoryRAG=memoryRAG,
                 #emotionRAG=emotionRAG,
                 vision_model="gemma3:4b",
                 huggingface_sentiment_model="cardiffnlp/twitter-roberta-base-sentiment",
                 huggingface_emotion_model="cardiffnlp/twitter-roberta-base-emotion",
                 name="Misato",
                 )
Brain.add_rag_system([us_histRAG,philRAG,mathRAG,italyRAG])
print("=================Prompting LLMR System===============")
#print(Brain.query("What is Jurisprudence?"))
#print(Brain.query("What is Representation Theory?"))
#print(Brain.query("Who was Karl Marx?"))
messages=[]
#emotion_history=[]
iterations=0
while True:
    os.system('ollama ps')
    try:    
        print("Listening")
        usr_in=input("==>")
        messages.append(("human",usr_in))
        print(usr_in)
        if(usr_in):
            print("Speaking")
            response=Brain.chat(messages)
            Speak(response)
            messages.append(("assistant",response))
            #emotion_history.append(("assistant",emotional_state))
            if len(messages)>6:
                print("Saving Messages to Disk")
                write_memories_to_disk(Brain.memoryRAG,messages)
                print("Memories Saved!")
                #print("Saving Emotional History to Disk")
                #write_memories_to_disk(Brain.emotionRAG,emotion_history)
                #print("Emotional History  Saved!")

                #emotion_history=[]
                messages=messages[2:]
    except Exception as e:
        print(e)
    iterations+=1
    iterations=iterations%10
    if(iterations==2):
        print("Compressing Memories")
        compress_memories(Brain.memoryRAG,messages)

        #print("Compressing Emotional History")
        #compress_memories(Brain.emotionRAG,emotion_history)
