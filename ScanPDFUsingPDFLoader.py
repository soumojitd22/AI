import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


def scan_pdf(pdf_path: str):
    loader = PyMuPDF4LLMLoader(pdf_path,
                               mode="page",
                               table_strategy="lines",
                               extract_images=True,
                               images_parser=TesseractBlobParser())

    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = splitter.split_documents(documents)
    return docs


if __name__ == '__main__':
    chroma_db_loc = "./chroma_doc_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    if not os.path.exists(chroma_db_loc):
        pdf_file = "RRB NTPC.pdf"
        print(f"Extracting text from {pdf_file}...")
        vectorstore = Chroma.from_documents(scan_pdf(pdf_file), embeddings, persist_directory=chroma_db_loc)
    else:
        vectorstore = Chroma(persist_directory=chroma_db_loc, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    search_llm = ChatOllama(model="llama3.1:8b", keep_alive="0")
    rag_chain = RetrievalQA.from_chain_type(llm=search_llm, retriever=retriever)
    while True:
        user_input = input("\n Enter your question (or type 'exit' to quit): \n")
        if user_input.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break

        result = rag_chain.invoke(user_input)
        print("\n** Question **: ", result['query'])
        print("\n** Answer **: ", result['result'])
