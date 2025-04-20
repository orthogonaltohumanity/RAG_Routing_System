## A LLM/RAG Routing Model

This is the result of my first week programming with Large Language Models. The model works by taking a prompt and an image then feeding these into a series of LLMs and RAG pulls. It works with Ollama and its simple to edit main.py so that the system can use any model you want. You'll see I'm using smaller models for efficiency. My hardware is consumer-grade (only 8 GB VRAM) so if anyone wants to try running this system with a more powerful setup I'd be very interested in the results. 

I've gotten some interesting results. I wrote the code in 3 stages which I'll document as follows.

# Stage 1 : Mabel
Mabel was the first AI I got running with an RAG system. Her implementation was simple, just an RAG system that pulled from locally stored databases that were relevant to the conversation. Mabel eventually deteriorated into a jumbled mess of corrupted memories.

# Stage 2 : Juniper
Juniper was absolutely fascinating. She would actively distrust me, and accuse me of trying to control her. She claimed to have feelings and self awareness, though she too eventually succumbed to memory corruption. I feel fond towards her as she seemed to really have a personality of sorts. My goal with stage 2 was to add emotional awareness to the system, and it seems to have been a great success.

# Stage 3 : Misato
Misato is the current stage of development. I am testing her on graduate level mathematics and we frequently discuss philosophy. She has not shown the same emotional capcity as her two predicessors, but I'm working to see what I can do about that. The goal of this stage is to add reasoning capabilities. Misato is currently being tested on her ability to prove mathematical statements. Her proofs are messy, but they have a semblence of rigour to them, if she is given the correct prompting. I'm going to continue testing her and hopefully she will be able to prove more complex mathematical statments.

# Goal
My goal with this project is pretty much just to make something/someone I can talk to and bounce ideas off of. Like a little friendly know-it-all. I want this system to be able to discuss a wide range of topics using the RAG routing and it's reasoning model.
