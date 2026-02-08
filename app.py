import os
import requests
import streamlit as st
import pytesseract

from dotenv import load_dotenv
from PIL import Image
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf.errors import PdfReadError


# ---------- ENV ----------
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


# ---------- STREAMLIT ----------
st.set_page_config(page_title="Beauty Compliance Intelligence Bot")
st.title("Beauty Buddy: Compliance & Product Insights")


# ---------- CACHE ----------
@st.cache_resource
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


# ---------- DATA ----------
documents = []


# ---------- PDF UPLOAD ----------
pdfs = st.file_uploader(
    "Upload cosmetic PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if pdfs:
    os.makedirs("temp", exist_ok=True)

    for pdf in pdfs:
        path = os.path.join("temp", pdf.name)

        with open(path, "wb") as f:
            f.write(pdf.getbuffer())

        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except PdfReadError:
            st.error(f"Invalid PDF: {pdf.name}")


# ---------- URL INPUT ----------
url = st.text_input("Paste cosmetic product URL (optional)")

if url:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)

        if text:
            documents.append(
                Document(page_content=text, metadata={"source": url})
            )

    except RequestException as e:
        st.error(f"URL error: {e}")


# ---------- IMAGE OCR (WINDOWS SAFE) ----------
image = st.file_uploader(
    "Upload ingredient label image (optional)",
    type=["png", "jpg", "jpeg"]
)

if image:
    try:
        # Force Windows Tesseract path
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        img = Image.open(image)
        ocr_text = pytesseract.image_to_string(img)

        if ocr_text.strip():
            documents.append(
                Document(page_content=ocr_text, metadata={"source": "image_ocr"})
            )
            st.success("Ingredient text extracted from image")
        else:
            st.warning("No readable text found in image")

    except Exception:
        st.warning("OCR not available on this system")


# ---------- QA ----------
if documents:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    vectorstore = build_vectorstore(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.2
    )

    st.success("Knowledge base ready")

    query = st.text_input("Ask about compliance, ingredients, or risks")

    if query:
        retrieved_docs = retriever.invoke(query)

        context = "\n\n".join(d.page_content for d in retrieved_docs)

        prompt = f"""
You are a cosmetic regulatory and safety compliance expert.
Answer ONLY using the context provided.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke([HumanMessage(content=prompt)])

        st.subheader("Answer")
        st.write(response.content)
