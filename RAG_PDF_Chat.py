import base64
import os
from io import BytesIO

from langchain_chroma import Chroma
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path


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


def transcribe_page(vision_llm, page_b64_png):
    response = vision_llm.invoke(build_ocr_prompt(page_b64_png))
    return response.content


def extract_text_from_pdf(pdf_path: str) -> str:
    vision_llm = ChatOllama(model="granite3.2-vision:latest", keep_alive="0")
    pages = pdf_pages_to_b64(pdf_path, dpi=400)
    texts = list(map(lambda page: transcribe_page(vision_llm, page), pages))
    return "\n\n".join(texts)


def extract_text_from_docs(docs: list[Document]) -> str:
    """
    Extracts text from a list of Document objects.

    Args:
        docs (list[Document]): List of Document objects.

    Returns:
        str: Concatenated text from all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def scan_pdf_locally(pdf_path: str) -> list[Document]:
    """
    Scans a PDF file and returns a list of Document objects.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[Document]: List of Document objects containing the scanned text.
    """
    loader = PyMuPDF4LLMLoader(pdf_path,
                               mode="page",
                               table_strategy="lines",
                               extract_images=True,
                               images_parser=TesseractBlobParser())

    documents = loader.load()
    return documents


def scan_pdf_and_embed(model_embeddings, chroma_db_loc) -> Chroma:
    """
    Scans a PDF file, extracts text using both OCR and local parsing, splits the text into chunks,
    and creates a Chroma vector store with the provided embeddings.

    Args:
        model_embeddings: Embedding model to use for the vector store.
        chroma_db_loc: Directory path to persist the Chroma vector store.

    Returns:
        Chroma: A Chroma vector store containing the embedded text chunks from the PDF.
    """
    pdf_file = "RRB NTPC.pdf"
    print(f"Extracting text from {pdf_file}...")
    extracted_text_from_ocr = extract_text_from_pdf(pdf_file)
    extracted_text_from_docs = extract_text_from_docs(scan_pdf_locally(pdf_file))
    text = extracted_text_from_ocr + "\n\n" + extracted_text_from_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(text)
    chroma_vectorstore = Chroma.from_texts(chunks, model_embeddings, persist_directory=chroma_db_loc)
    return chroma_vectorstore


def get_retriever() -> Chroma.as_retriever:
    """
    Initializes and returns a retriever object for searching over a Chroma vector store.
    If the Chroma database does not exist, it scans and embeds the PDF, otherwise it loads the existing vector store.

    Returns:
        BaseRetriever: A retriever for querying the vector store.
    """
    chroma_db_loc = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    if not os.path.exists(chroma_db_loc):
        vectorstore = scan_pdf_and_embed(embeddings, chroma_db_loc)
    else:
        vectorstore = Chroma(persist_directory=chroma_db_loc, embedding_function=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def generate_chain(context_retriever):
    """
    Generates a chain for question answering using a retriever and a language model.

    Args:
        context_retriever: The retriever to use for fetching context.

    Returns:
        Chain: A chain that processes user input and generates answers.
    """
    search_llm = ChatOllama(model="llama3.1:8b", keep_alive="0")
    system_prompt = (
        """
        You are a helpful AI assistant.
        You can answer questions based on the provided context.
        Also you can answer general questions that does not require context.
        Your answer should be crisp, concise and to-the-point.
        Context: {context}
        """
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}")
    ])

    search_chain = (
            {"context": context_retriever, "input": RunnablePassthrough()}
            | prompt_template
            | search_llm
            | StrOutputParser()
    )

    return search_chain


def interactive_chat():
    retriever = get_retriever()
    chain = generate_chain(retriever)
    while True:
        user_input = input("\nEnter your question (or type 'exit' to quit): \n")
        if user_input.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break

        llm_response = chain.invoke(user_input)
        print("\n**Answer**: \n", llm_response)


if __name__ == "__main__":
    interactive_chat()
