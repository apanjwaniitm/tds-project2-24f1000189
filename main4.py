from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import fitz
from typing import Optional

# Get the AIPROXY_TOKEN from environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set! Run 'export AIPROXY_TOKEN=your-token' before starting FastAPI.")

# LLM Proxy endpoint (update as needed)
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

app = FastAPI(title="Enhanced TDS Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class AnswerResponse(BaseModel):
    answer: str

# A detailed system prompt for improved response behavior
SYSTEM_PROMPT = (
    "You are a highly accurate data science assignment assistant. "
    "Before providing your final answer, think step by step through the problem, "
    "double-check any calculations, and only output the final answer without extra commentary."
    "While answering, do not return any additional text. If the answer is a number, return ONLY a number."
    "If the answer is a text, return ONLY the precise text answering the question."
)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF as a single string.
    Only the first 3000 characters are returned to avoid oversized prompts.
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in pdf_document)
        return text[:3000]
    except Exception as e:
        print("Error extracting text from PDF:", e)
        return ""

def extract_text_from_jsonl(jsonl_bytes: bytes) -> str:
    """
    Extract text from a JSONL (or JSON) file.
    Returns the first 3000 characters.
    """
    try:
        text = jsonl_bytes.decode("utf-8", errors="ignore")
        return text[:3000]
    except Exception as e:
        print("Error extracting text from JSONL:", e)
        return ""

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncates the text if it exceeds a maximum number of characters.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."

def build_enhanced_prompt(question: str, context: str) -> str:
    """
    Builds a prompt that instructs the LLM to use chain-of-thought reasoning.
    It incorporates an optional context from an attached file.
    """
    prompt = "Below is the chain-of-thought reasoning process:\n"
    if context:
        prompt += f"Context:\n{context}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Now, think step-by-step through the problem and then provide only the final answer."
    return prompt

def get_llm_answer(question: str, context: str = "") -> str:
    """
    Calls the LLM using a detailed prompt with chain-of-thought instructions.
    Adjusts parameters like temperature and max_tokens for more deterministic output.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    # Adjust lengths as needed. Here we make sure neither question nor context is excessively long.
    MAX_QUESTION_LENGTH = 2500
    truncated_question = truncate_text(question, MAX_QUESTION_LENGTH)
    truncated_context = truncate_text(context, 3000) if context else ""
    
    prompt = build_enhanced_prompt(truncated_question, truncated_context)
    
    payload = {
        "model": "gpt-4o-mini",  # Change this if you have access to a different model variant.
        "temperature": 0,        # Lower temperature for deterministic responses.
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    }
    
    try:
        response = httpx.post(AIPROXY_BASE_URL, json=payload, headers=headers, timeout=60.0)
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"].strip()
            return reply
        elif response.status_code == 429:
            return "AIPROXY monthly request limit reached. Try again next month!"
        elif response.status_code == 403:
            return "AIPROXY cost limit exceeded. Requests are blocked!"
        else:
            return f"AIPROXY Error {response.status_code}: {response.text}"
    except httpx.TimeoutException:
        return "AIPROXY request timed out. Try again later."
    except httpx.RequestError as e:
        return f"Failed to connect to AIPROXY: {str(e)}"

@app.post("/api/", response_model=AnswerResponse)
async def solve_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    This endpoint receives an assignment question and an optional file.
    It extracts text (for PDFs, JSONL, etc.), builds an enhanced prompt with chain-of-thought instructions, and returns the LLM answer.
    """
    context = ""
    if file:
        filename = file.filename.lower()
        try:
            file_bytes = await file.read()
            if filename.endswith(".pdf") or file.content_type == "application/pdf":
                context = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".jsonl") or filename.endswith(".json"):
                context = extract_text_from_jsonl(file_bytes)
            else:
                # If the file is of other type, try a simple text decode.
                context = file_bytes.decode("utf-8", errors="ignore")[:3000]
        except Exception as e:
            print(f"Error reading file {file.filename}: {str(e)}")
    
    answer = get_llm_answer(question, context)
    return AnswerResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)