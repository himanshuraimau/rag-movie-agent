import os
from typing import List
from langchain_community.document_loaders import CSVLoader  # Changed to CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def load_and_process_data(file_path: str):
    loader = CSVLoader(file_path)  # Changed to CSVLoader
    documents = loader.load()
    
    # Limit to first 100 rows
    documents = documents[:100]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def setup_vector_store(chunks: List):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="netflix_data",
        persist_directory="chroma_db"
    )

def setup_llm():
    return ChatOllama(
        model="llama3.2:1b",
        temperature=0.7,
        num_predict=2048,
        top_k=10,
        top_p=0.95,
        repeat_penalty=1.1,
    )

def main():
    file_path = "/home/himanshu/code/agents/rag_movie_agent/knowledge/netflix_titles.csv"
    chunks = load_and_process_data(file_path)
    vectorstore = setup_vector_store(chunks)
    llm = setup_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    print("System is ready for queries!")
    
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        try:
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:", result["result"])
            print("\nSource Documents:")
            for doc in result.get("source_documents", []):
                print(f"- {doc.page_content[:200]}...")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
