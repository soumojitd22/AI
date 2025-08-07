import base64
import os
from io import BytesIO

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from pdf2image import convert_from_path

from ScanPDFUsingPDFLoader import scan_pdf


def pdf_pages_to_b64(pdf_path: str, dpi=300):
    """
    Converts each page of a PDF file to a base64-encoded PNG image.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int, optional): Dots per inch for image conversion. Defaults to 300.

    Returns:
        list[str]: List of base64-encoded PNG images, one for each page.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    print("Page count: ", len(images))
    encoded = []
    for i, page_img in enumerate(images, 1):
        buf = BytesIO()
        aspect_ratio = page_img.height / page_img.width
        target_width = 2000
        target_height = int(target_width * aspect_ratio)
        resized_img = page_img.resize((target_width, target_height))
        resized_img.save(buf, format="PNG", optimize=True, quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        encoded.append(b64)

    return encoded


def build_ocr_prompt(b64_png: str) -> list:
    """
    Builds a prompt for an OCR task using a base64-encoded PNG image.

    Args:
        b64_png (str): Base64-encoded PNG image string.

    Returns:
        list: A list containing a HumanMessage with image and instruction for OCR extraction.
    """
    return [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                },
                {
                    "type": "text", "text":
                    "You are an expert OCR engine. "
                    "This is a scanned page from a PDF document. This image has data in tabular format mostly."
                    "I want the text extracted from the tables inside the image."
                }
            ]
        )
    ]


def transcribe_page(b64_png):
    response = llm.invoke(build_ocr_prompt(b64_png))
    return response.content


def extract_text_from_pdf(pdf_path: str) -> str:
    pages = pdf_pages_to_b64(pdf_path, dpi=400)
    texts = list(map(transcribe_page, pages))
    return "\n\n".join(texts)


if __name__ == "__main__":
    chroma_db_loc = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    if not os.path.exists(chroma_db_loc):
        llm = ChatOllama(model="granite3.2-vision:latest", keep_alive="0")
        pdf_file = "RRB NTPC.pdf"
        print(f"Extracting text from {pdf_file}...")
        extracted_text = extract_text_from_pdf(pdf_file)
        splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
        chunks = splitter.split_text(extracted_text)
        vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=chroma_db_loc)
        vectorstore.add_documents(scan_pdf(pdf_file))
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
