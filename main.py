# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import base64
import pytesseract # For Tesseract setup if needed

# Import document parsing and embedding utilities
from document_parser import parse_document, parse_image_ocr
from embedding_utils import vector_store # Assuming vector_store is a global instance

# Import OpenAI client
import openai

# Load environment variables from .env file
load_dotenv()

# --- IMPORTANT: Configure Tesseract path if it's not in your system's PATH ---
# Uncomment and set this path if you encounter "TesseractNotFoundError"
# For Windows example:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS example (often not needed if installed via Homebrew):
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Configure OpenAI API key
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    # Initialize OpenAI client (for newer openai library versions)
    openai_client = openai.OpenAI(api_key=openai_api_key)
except ValueError as e:
    print(f"Configuration Error: {e}")
    openai_client = None # Set to None if API key is not available

app = FastAPI(
    title="Smart RAG API",
    description="A Retrieval-Augmented Generation API for various document types.",
    version="0.1.0"
)

# In-memory store for tracking uploaded file_ids (metadata, not actual chunks)
uploaded_files_info: Dict[str, Any] = {}
next_file_id = 1

# Define a simple health check endpoint
@app.get("/", summary="Health check endpoint")
async def read_root():
    return {"message": "Smart RAG API is running!"}

# Document Ingestion Endpoint (from Step 2, unchanged)
@app.post("/upload/", summary="Uploads a document for processing and returns a file_id")
async def upload_document(file: UploadFile = File(..., description="The document file to upload (e.g., .pdf, .docx, .txt, .jpg, .png, .csv, .db)")):
    global next_file_id
    current_file_id = f"doc_{next_file_id}"
    next_file_id += 1

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected or filename is empty."
        )

    try:
        file_content = await file.read()

        text_chunks = parse_document(file_content, file.filename)

        if not text_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract meaningful text chunks from the document."
            )

        # Add chunks to the vector store (from Step 3)
        vector_store.add_chunks(doc_id=current_file_id, chunks=text_chunks, filename=file.filename)

        uploaded_files_info[current_file_id] = {
            "filename": file.filename,
            "num_chunks": len(text_chunks),
            "file_type": file.content_type,
            "size_bytes": len(file_content),
            "status": "Processed and added to vector store"
        }

        return {
            "file_id": current_file_id,
            "filename": file.filename,
            "status": "Document processed and indexed successfully",
            "num_chunks": len(text_chunks),
            "first_chunk_sample": text_chunks[0][:200] + "..." if text_chunks else ""
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document upload or processing: {e}"
        )

# Endpoint to list uploaded files (useful for testing and debugging)
@app.get("/files/", summary="Get information about currently uploaded and indexed files")
async def get_uploaded_files_summary():
    # Provide a summary of documents that have been added to the vector store
    return {"indexed_documents_count": len(uploaded_files_info), "files": uploaded_files_info}


# --- LLM Utility Function ---
async def get_llm_response(query: str, context: str) -> str:
    """
    Sends the query and context to the LLM and returns the generated answer.
    """
    if openai_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key is not configured. Cannot generate LLM response."
        )

    system_message = (
        "You are a helpful AI assistant tasked with answering questions based *only* on the provided context. "
        "If the answer cannot be found in the context, state that you don't have enough information. "
        "Do not make up information. Be concise and to the point."
    )

    user_message = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        chat_completion = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4", "gpt-4-turbo-preview", etc., based on availability and cost
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7, # Controls randomness: 0.0 for deterministic, 1.0 for more creative
            max_tokens=500 # Limit response length
        )
        return chat_completion.choices[0].message.content.strip()
    except openai.APIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API error: {e.status_code} - {e.response}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during LLM interaction: {e}"
        )


# Query Endpoint (Updated for Step 4)
class QueryRequest(BaseModel):
    question: str
    image_base64: str = None # Optional: Base64 encoded image for image-based questions

class SourceInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    # You could add 'page_number' here if your parser extracted it and stored it in metadata

class QueryResponse(BaseModel):
    question: str
    retrieved_context: List[str]
    source_info: List[SourceInfo]
    final_answer: str

@app.post("/query/", response_model=QueryResponse, summary="Queries the uploaded documents based on the question and generates an answer using an LLM.")
async def query_document(request: QueryRequest):
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been indexed yet. Please upload a document first using the /upload endpoint."
        )

    query_text = request.question

    # Handle image-based questions (OCR)
    if request.image_base64:
        try:
            image_bytes = base64.b64decode(request.image_base64)
            ocr_text = parse_image_ocr(image_bytes)
            if ocr_text:
                query_text = f"{query_text}\n\n[Content from image]: {ocr_text}"
            else:
                print(f"Warning: OCR could not extract text from provided image for query: {request.question[:50]}...")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing image_base64 for OCR: {e}"
            )

    # Perform vector search (from Step 3)
    # Retrieve top N relevant chunks. You can adjust 'k' based on desired context length.
    relevant_chunks_info = vector_store.search(query=query_text, k=5)

    if not relevant_chunks_info:
        # If no relevant chunks are found, inform the user/LLM.
        # We can still pass this to LLM to say it doesn't have info, or return early.
        # For now, let's return a specific message.
        return QueryResponse(
            question=request.question,
            retrieved_context=[],
            source_info=[],
            final_answer="I could not find relevant information in the uploaded documents to answer your question."
        )

    retrieved_context_texts = [chunk_info["text"] for chunk_info in relevant_chunks_info]
    source_info_list = [
        SourceInfo(
            doc_id=chunk_info["doc_id"],
            filename=uploaded_files_info.get(chunk_info["doc_id"], {}).get("filename", "unknown"), # Use uploaded_files_info for filename
            chunk_index=chunk_info["chunk_index"]
        ) for chunk_info in relevant_chunks_info
    ]

    # Combine retrieved context for the LLM prompt
    # Use a clear separator between chunks
    combined_context = "\n\n---\n\n".join(retrieved_context_texts)

    # Call the LLM with the question and combined context
    final_answer = await get_llm_response(query_text, combined_context)

    return QueryResponse(
        question=request.question,
        retrieved_context=retrieved_context_texts,
        source_info=source_info_list,
        final_answer=final_answer
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)