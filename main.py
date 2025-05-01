import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import glob
from collections import defaultdict


def load_chat_history(user_name):
    
    full_path = f"conversation_history/{user_name}.txt"
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            return file.read() 
    return ""

def save_chat_history(history, user_name):

    full_path = f"conversation_history/{user_name}.txt"
    with open(full_path, 'w', encoding='utf-8') as file:
        file.write(history)


def run_ollama(prompt: str):

    result = subprocess.run(
        ["ollama", "run", "llama3.2"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )
    return result.stdout

def build_persona_knowledge_base(data_of_documents: list):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data_of_documents)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index

def attain_context(query, model, index, documents, k=3):

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=k)
    retrieved_docs = [documents[i] for i in indices[0]]

    return retrieved_docs

def retrieve_persona_data(folder_path: str) -> list:

    persona_data = []

    for file_path in glob.glob(os.path.join(folder_path, "*")):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.read()
                persona_data.append(data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return persona_data

def chat_with_persona(user_query,model, index, data_of_documents, user_name="friend", history=None):
    retrieved_docs = attain_context(user_query, model, index, data_of_documents)
    context = " ".join(retrieved_docs)

    history_text = "\n".join(history) if history else ""

    prompt = f"""
    You are now {context}. You have the following personality, interests, and history:
    {context}

    Conversation history:
    {history}

    Here is the ongoing conversation:
    {history}

    Continue the conversation naturally, maintaining the tone, personality, and flow of the conversation. Do not restart the conversation, repeat questions, or add unnecessary greetings. Respond as if you are this person, using their tone of voice and personality. 

    Your response should feel personal, warm, and conversational, as if speaking to a close friend, family member, or someone familiar. Use humor, emotions, and relatable expressions where appropriate. Avoid sounding robotic or overly formal.

    Question: {user_query}
    Answer:
    """
    response = run_ollama(prompt)
    return response


def main():


    user_name = input('What is your name?  ').strip()
    history = load_chat_history(user_name)
    recent_history = ''

    data_of_documents = retrieve_persona_data('persona_data')
    model, index = build_persona_knowledge_base(data_of_documents)

    persona_name = 'Mares'
    user_query = input(f"{persona_name}: Hiii! \n\n{user_name}: ")

    while user_query.lower() != "bye":

        history += f"\n{user_name}: {user_query}\n"

        response = chat_with_persona(user_query,model, index, data_of_documents,user_name, history)
        history += f"{persona_name}: {response}\n"
        
        user_query = input(f"\n{persona_name}: {response}\n{user_name}: ").strip()
    
    save_chat_history(history, user_name)
    


if __name__ == "__main__":
    main()