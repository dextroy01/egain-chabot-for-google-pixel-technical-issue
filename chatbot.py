"""Imports and Global Variables"""
from openai import OpenAI
import PyPDF2
import os
import numpy as np
import faiss
import logging

#Silences errors PyPDF throws when parsing pdf's
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

client: OpenAI
conversation: dict

""" Extract_chunks_from_manuals()
    Extracts text from each pdf in the user manual folder and turns it into chunks

    inputs - folder_path: string, chunk size: int
    outputs - chunks: list[str] (a list of all the text from the pdf split into chunks) 
"""
def extract_chunks_from_manuals(folder_path, chunk_size=300):
    chunks = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        with open(os.path.join(folder_path, filename), "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                # Chunk page text into 300-word chunks
                words = text.split()
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    chunks.append(f"{filename}, page {page_num+1}:\n{chunk}")
    return chunks


""" get_embedding()
    Uses openAi's embedding api to convert a chunk into its vector representation

    input - text: string, model: string (which openAi embedding model to use)
    output - np.array (a chunks vector repsentation)
"""
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


""" build_index()
    Takes a list of text chunks, generates embeddings, and stores them in a FAISS index for fast vector similarity search

    inputs - chunks: list[str]
    outputs - index:faiss.IndexFlatL2 (a vectorized database of user manuals for the model), embeddings: List[np.ndarray]
"""
def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings


""" retrieve_relevant chunks()    
    Searches the FAISS index to find the top k most relevant chunks for the user's question.

    inputs: query: String, chunks:list[str], index:faiss.IndexFlatL2, k:int (the top k most relevant chunks)
    outputs: combined_chunks: list[str] (The most relevant chunks combined together as a string)
"""
def retrieve_relevant_chunks(query, chunks, index, k=3):
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]), k)

    #only uses the top 20 chunks so the input size isnt too long for the openAi api.
    combined_chunks = "".join(chunks[:20])
    return combined_chunks


""" initilize_chatbot_and_vector_databass()
    Ssks the user for api key to use openAi's api. If key is incorrect throws an error.
    Parses the pdf's and creates the vector database

    inputs - none
    ouput - chunks: list[str] (a list of all the text from the pdf split into chunks),  index:faiss.IndexFlatL2 (a vectorized database of user manuals for the model)
"""
def initilize_chatbot_and_vector_databass():
    global client
    global conversation

    #Asks user for api key and throw an error if key is inncorrect, checks if key is correct by sending a request to openAi's embedding model
    while True:
        api_key = input("\nPlease input the API key to initilize the chatbot attached in the email and slides:\n\n" \
                        "You: ")
        client = OpenAI(api_key = api_key)
        try:
            client.embeddings.create(model="text-embedding-ada-002", input="ping")
            break
        except Exception as e:
            print("Incorrect API key, please see email or slides with submission\n"
                  "Try again\n")

    #the system prompt to tell the model its purpose
    system_prompt = "Your purpose is to guide a customer through troubleshooting an issue with there pixel 9 and pixel 9 pro fold"
    conversation = [{"role": "system", "content": system_prompt}]
    
    #extract chunks from userManuals and create a vector database
    print("\nThank you, please wait while we parse user manuals and generate a vector database\n")
    chunks = extract_chunks_from_manuals("userManuals/")
    index, _ = build_index(chunks)

    return chunks, index

""" get_user_issue()
    Asks the user what error they are expiriencing and which phone model they need help with. 
    Catches error if the user asks about a phone that this model is not trained to deal with and asks again

    inputs - None
    outputs user_issue: String(What product the user is struggling with and the issue it has)
"""
def get_user_issue():
    #Get the product and issue the user needs help with
    user_issue = ""
    while True:
        user_product = input("eDexter: Please input what product you need help with Pixel 9 or Pixel 9 Pro Fold\n\n" 
                         "You: ")
        if user_product.lower() in ["pixel 9", "pixel 9 pro fold"]:
            break
        else:
            print("\neDexter: Product is not supported by this chatbot, try again\n")
    technical_issue = input("\neDexter: Please input what technical issue you are having with the product. (i.e not charing, won't turn on, or cracked screen)\n\n"
                            "You: ")

    user_issue += "the product " + user_product + " has this issue " + technical_issue

    return user_issue 

""" get_chatbot_reply()
    calls openAi's api and prints the chabots response to a prompt

    input - none
    output chatbot_reply: string
"""
def get_chatbot_reply():
    # I use a low temperature to make sure the chatbot is more deterministic
    global conversation

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = conversation,
        temperature = 0.3,
        stream= True)

    chatbot_reply = ""

    #Prints the chatbot's response as it is created
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end = "", flush = True)
            chatbot_reply += content

    print()

    return chatbot_reply

""" chat()
    gets input from user and passes it to the chatbot. will exit chat if prompted

    input - chunk, index
    output - None
"""
def chat(chunks, index):
    print("Hello I am your Custom Service Chatbot eDexter\n"
          "Please input the product you need help with.\n"
          "When you are finished, please type \"Exit\"\n"
          "If you need help with another product type \"New Issue\"\n")
    
    global conversation
    running = True

    #Gets the issue from user and prompts the chatbot for a response. 
    while running:
        user_issue = get_user_issue()
        user_input = ""
        top_chunks = retrieve_relevant_chunks(user_issue, chunks, index)

        conversation.append({"role": "user", 
                             "content": "here is the user issue" + user_issue + "and the relevant chunks from the user manual" + top_chunks})

        while True: 
            #if the user inputs "new issue" then the chat starts over
            if user_input.lower() in ["new issue"]:
                break
            
            #if the user inputs exit the code exits
            if user_input.lower() in ["exit"]:
                print("\neDexter: Goodbye, Thank you for using our service!")
                running = False
                break
                
            print("\neDexter: ", end = "", flush = True)
            chatbot_reply = get_chatbot_reply()
                
            conversation.append({"role": "assistant", "content": chatbot_reply})
            
            user_input = input("\nYou: ")
            relevant_chunks = retrieve_relevant_chunks(user_issue, chunks, index, k=3)
            conversation.append({"role": "user", 
                                 "content": "the input is " + user_input + " and here are the relevant chunks from the user manual " + relevant_chunks })


if  __name__== "__main__":
    chunks, index = initilize_chatbot_and_vector_databass()
    chat(chunks, index)

