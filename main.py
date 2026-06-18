from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag import extract_text, split_text, store_chunks, ask_question
import shutil
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text(file_path)
    chunks = split_text(text)
    store_chunks(chunks)
    os.remove(file_path)

    return {"message": f"{len(chunks)} chunk stored."}

@app.post("/ask")
async def ask(request: QuestionRequest):
    answer = ask_question(request.question)
    return {"answer": answer}

@app.post("/reset")
async def reset():
    from rag import reset_collection
    reset_collection()
    return {"message": "ChromaDB resetted."}
