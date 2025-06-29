from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
import os
import shutil
from chromadb.config import Settings

settings = Settings(anonymized_telemetry=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "chroma_db"
qa = None

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    filepath = f"temp_{file.filename}"
    with open(filepath, "wb") as f:
        f.write(contents)

    loader = PyPDFLoader(filepath)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        client_settings=settings
    )
    vectordb.persist()

    global qa
    llm = LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
        n_ctx=2048,
        verbose=True
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return {"message": "PDF processed"}

@app.post("/ask/")
async def ask_question(q: dict):
    global qa
    if not qa:
        return {"error": "Upload a PDF first"}
    return {"answer": qa.run(q["question"])}