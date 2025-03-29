from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import fitz  # PyMuPDF for PDF processing
import json
from typing import Optional
from PIL import Image
from io import BytesIO

# Get API Token from Environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set! Run 'export AIPROXY_TOKEN=your-token' before starting FastAPI.")

# LLM Proxy endpoint (update if needed)
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

app = FastAPI(title="Enhanced File Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Model
class AnswerResponse(BaseModel):
    answer: str

UPLOAD_FOLDER = "processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# System Prompt for LLM
SYSTEM_PROMPT = (
    "You are an advanced assistant that processes uploaded files and answers queries. "
    "Use provided context if available. Return only the precise answer."
)

# --------------- TEXT FILE PROCESSING --------------- #

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF file"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in pdf_document)
        return text[:3000]  # Limit to first 3000 characters
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_json(json_bytes: bytes) -> str:
    """Extracts text from a JSON file"""
    try:
        data = json.loads(json_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2)[:3000]  # Convert JSON to string, limit to 3000 characters
    except Exception as e:
        print(f"Error extracting text from JSON: {e}")
        return ""

def extract_text_from_txt(txt_bytes: bytes) -> str:
    """Extracts text from a TXT file"""
    try:
        return txt_bytes.decode("utf-8", errors="ignore")[:3000]
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# --------------- IMAGE PROCESSING --------------- #

def process_image(image_bytes: bytes, filename: str) -> str:
    """Processes an image and returns the path to the processed file"""
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")  # Ensure standard format

        # Save reconstructed image
        output_path = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
        image.save(output_path, format="PNG")

        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

# --------------- LLM QUERY --------------- #

def get_llm_answer(question: str, context: str = "") -> str:
    """Calls the LLM using an enhanced prompt with extracted context."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide the most accurate answer."

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    }

    try:
        response = httpx.post(AIPROXY_BASE_URL, json=payload, headers=headers, timeout=60.0)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error {response.status_code}: {response.text}"
    except httpx.RequestError as e:
        return f"Failed to connect to LLM: {e}"

# --------------- FASTAPI ENDPOINT --------------- #

@app.post("/api/", response_model=AnswerResponse)
async def process_file(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts a question and an optional file. Processes the file based on type
    and returns either extracted text or a reconstructed image.
    """
    if not file:
        return AnswerResponse(answer=get_llm_answer(question))

    filename = file.filename.lower()
    file_bytes = await file.read()
    
    # Determine file type and extract content
    if filename.endswith(".pdf") or file.content_type == "application/pdf":
        context = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".json") or filename.endswith(".jsonl"):
        context = extract_text_from_json(file_bytes)
    elif filename.endswith(".txt"):
        context = extract_text_from_txt(file_bytes)
    elif filename.endswith((".png", ".jpeg", ".jpg", ".webp")):
        output_path = process_image(file_bytes, filename)
        if output_path:
            return FileResponse(output_path, media_type="image/png", filename=f"processed_{filename}")
        else:
            raise HTTPException(status_code=500, detail="Error processing image")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # If text was extracted, pass it to LLM
    answer = get_llm_answer(question, context)
    return AnswerResponse(answer=answer)