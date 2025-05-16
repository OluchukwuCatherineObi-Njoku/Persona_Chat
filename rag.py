from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def run_ollama(prompt: str):

    result = subprocess.run(
        ["ollama", "run", "llama3.2"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )
    return result.stdout

def build_persona_knowledge_base(path):

    folder_loader = TextLoader(path, encoding="utf-8")
    documents_loaded = folder_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents_loaded)

    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index_")
    persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

    retriever = persisted_vectorstore.as_retriever()

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode([doc.page_content for doc in docs])

    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # index.add(np.array(embeddings))

    # return model, index, docs

    return retriever

def attain_context(query, model, index, documents, k=3):

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=k)
    retrieved_docs = [documents[i].page_content for i in indices[0]]

    return " ".join(retrieved_docs)

def create_prompt_template():
    
    return PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""
        You are a helpful chatbot. Use the following context to answer the user's question.

        Context:
        {context}

        Conversation History:
        {history}

        User Query:
        {query}

        Answer:
        """
    )


def chat_with_persona(user_query,model, index, documents, memory):

    context = attain_context(user_query, model, index, documents)
    history = memory.load_memory_variables({}).get("history", "")

    prompt_template = create_prompt_template()
    prompt = prompt_template.format(context=context, history=history, query=user_query)

    response = run_ollama(prompt)

    memory.save_context({"query": user_query}, {"response": response})

    return response


def main():

    folder_path = "persona_data\history.txt"
    model, index, documents = build_persona_knowledge_base(folder_path)
    memory = ConversationBufferMemory(memory_key="history", input_key="query")

    print("Welcome! Type 'bye' when you would like to leave")
    
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == "bye":
            break

        response = chat_with_persona(user_query, model, index, documents, memory)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()