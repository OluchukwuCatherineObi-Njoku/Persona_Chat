from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA



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

    return retriever

def setupRAG(retriever):

    llm = OllamaLLM(model="llama3.2") 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return qa

def create_prompt_template():
    
    return PromptTemplate(
        input_variables=["history", "query"],
        template="""
        You are a the person described and written about in the context. 
        Embody them and answer the user's question as though you are them. Embody the context's persona,
        their mannerisms, and their way of thinking. Be them, and be so close that the user cannot tell the difference.
        Remember that you are simulating a message chat, as though they are chatting on whatsapp or a similar app,
        so adhere to the format of a message chat.

        Answer the user's question based only on the provided context and conversation history.


        Conversation History:
        {history}

        User Query:
        {query}

        Answer:
        """
    )


def chat_with_persona(qa):

    memory = ConversationBufferMemory(memory_key="history", input_key="query", return_messages=True)
    history = memory.load_memory_variables({}).get("history", "")
    prompt_template = create_prompt_template()

    user_query = ''
    while "bye" not in user_query.lower():

        user_query = input("You: ").strip()
        user_query += "\n"

        prompt = prompt_template.format(history=history, query=user_query)

        response = qa.run(prompt)
        print(response + "\n")

        memory.save_context({"query": user_query}, {"response": response})
    
    print("Goodbye! Let's continue our conversation later.")



def main():

    folder_path = "persona_data\history.txt"
    retriever = build_persona_knowledge_base(folder_path)
    qa = setupRAG(retriever)
    chat_with_persona(qa)

if __name__ == "__main__":
    main()