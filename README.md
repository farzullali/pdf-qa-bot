# pdf-qa-bot

This project allows you to ask questions about an uploaded PDF using an LLM.

## Stack

FastAPI, ChromaDB, LangChain, Sentence Transformers, Claude API

## How it works

After uploading a PDF, the text is extracted and split into chunks. Each chunk is embedded and stored in ChromaDB. When a question is asked, it is also embedded and the 3 most relevant chunks are retrieved. These chunks are sent to the LLM along with the question to generate an answer.

## Usage

- `POST /upload` — upload a PDF file
- `POST /ask` — ask a question about the PDF
- `POST /reset` — clear ChromaDB
