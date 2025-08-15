import base64
import logging
import os
from io import BytesIO
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (should be moved to config or environment variables in production)
DOC_SCANNER_MODEL: str = os.getenv("DOC_SCANNER_MODEL", "granite3.2-vision:latest")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
QUESTION_ELABORATOR_MODEL: str = os.getenv("QUESTION_ELABORATOR_MODEL", "mistral:7b")
SEARCH_MODEL: str = os.getenv("SEARCH_MODEL", "llama3.1:8b")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_FILE: str = os.getenv("PDF_FILE", "RRB NTPC.pdf")


def pdf_pages_to_b64(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    Converts each page of a PDF file to a base64-encoded PNG image.
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise

    logger.info(f"Page count: {len(images)}")
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


def build_ocr_prompt(b64_png: str) -> List[HumanMessage]:
    """
    Builds a prompt for an OCR task using a base64-encoded PNG image.
    """
    return [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                },
                {
                    "type": "text",
                    "text": (
                        "You are an expert OCR engine. "
                        "This is a scanned page from a PDF document. This image has data in tabular format mostly."
                        "I want to get the extracted text in tabular format in which format it is there in the image."
                    ),
                },
            ]
        )
    ]


def transcribe_page(vision_llm: ChatOllama, page_b64_png: str) -> str:
    try:
        response = vision_llm.invoke(build_ocr_prompt(page_b64_png))
        return response.content
    except Exception as e:
        logger.error(f"OCR transcription failed: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    vision_llm = ChatOllama(model=DOC_SCANNER_MODEL, keep_alive="0")
    try:
        pages = pdf_pages_to_b64(pdf_path, dpi=400)
        texts = [transcribe_page(vision_llm, page) for page in pages]
        return "\n\n".join(texts)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""


def extract_text_from_docs(docs: List[Document]) -> str:
    """
    Extracts text from a list of Document objects.
    """
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)


def scan_pdf_locally(pdf_path: str) -> List[Document]:
    """
    Scans a PDF file and returns a list of Document objects.
    """
    try:
        loader = PyMuPDF4LLMLoader(
            pdf_path,
            mode="page",
            table_strategy="lines",
            extract_images=True,
            images_parser=TesseractBlobParser(),
        )
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Failed to scan PDF locally: {e}")
        return []


def scan_pdf_and_embed(model_embeddings: OllamaEmbeddings, pdf_file: str = PDF_FILE) -> Chroma:
    """
    Scans a PDF file, extracts text using both OCR and local parsing, splits the text into chunks,
    and creates a Chroma vector store with the provided embeddings.
    """
    logger.info(f"Extracting text from {pdf_file}...")
    extracted_text_from_ocr = extract_text_from_pdf(pdf_file)
    extracted_text_from_docs = extract_text_from_docs(scan_pdf_locally(pdf_file))
    text = f"{extracted_text_from_ocr}\n\n{extracted_text_from_docs}"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(text)
    chroma_vectorstore = Chroma.from_texts(chunks, model_embeddings, persist_directory=CHROMA_DB_PATH)
    return chroma_vectorstore


def get_retriever(pdf_file: str = PDF_FILE) -> Chroma.as_retriever:
    """
    Initializes and returns a retriever object for searching over a Chroma vector store.
    """
    chroma_db_loc = CHROMA_DB_PATH
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists(chroma_db_loc) or not os.listdir(chroma_db_loc):
        logger.info("Chroma DB not found or empty. Creating new vector store...")
        vectorstore = scan_pdf_and_embed(embeddings, pdf_file)
    else:
        logger.info("Loading existing Chroma vector store...")
        vectorstore = Chroma(persist_directory=chroma_db_loc, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def get_search_prompt_template() -> ChatPromptTemplate:
    """
    Creates a prompt template for searching context in a question-answering system.
    """
    system_prompt = (
        "You are a helpful AI assistant.\n"
        "You can answer questions based on the provided context.\n"
        "Also you can answer general questions that does not require context.\n"
        "Your answer should be crisp, concise and to-the-point.\n"
        "Context: {context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}")
    ])


def get_elaborator_prompt_template() -> ChatPromptTemplate:
    """
    Creates a prompt template for elaborating on questions in a question-answering system.
    """
    system_prompt = (
        "You are a helpful AI assistant.\n"
        "You can elaborate the input question to make it more specific, clear, crisp, concise and to-the-point question.\n"
        "Do not answer the question, just elaborate the question and generate a question only.\n"
        "If the question asks about 'candidate', elaborate to 'applicant in the context'.\n"
        "Example: \"What is the name of the candidate?\" should be\n"
        "\"What is the name of the applicant?\"\n"
        "Example: \"What is the registration number of the candidate?\" should be\n"
        "\"What is the registration number of the applicant?\""
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


def generate_chain(context_retriever) -> Runnable:
    """
    Generates a chain for question answering using a retriever and a language model.
    """
    elaborator_llm = ChatOllama(model=QUESTION_ELABORATOR_MODEL, keep_alive="0")
    search_llm = ChatOllama(model=SEARCH_MODEL, keep_alive="0")

    def log_and_return(label, value):
        logger.info(f"{label}: {value}")
        return value

    search_chain = (
        get_elaborator_prompt_template()
        | elaborator_llm
        | StrOutputParser()
        | (lambda question: log_and_return("Elaborated Question", question))
        | {"context": context_retriever, "input": RunnablePassthrough()}
        | get_search_prompt_template()
        | (lambda prompt: log_and_return("Prompt", prompt))
        | search_llm
        | StrOutputParser()
    )
    return search_chain


def interactive_chat():
    retriever = get_retriever()
    chain = generate_chain(retriever)
    logger.info("Interactive chat started. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nEnter your question (or type 'exit' to quit): \n")
            if user_input.lower() == 'exit':
                logger.info("Exiting the chat. Goodbye!")
                break
            llm_response = chain.invoke(user_input)
            print("\n**Answer**: \n", llm_response)
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")


if __name__ == "__main__":
    interactive_chat()