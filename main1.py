import multipart
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
import os
import httpx
from typing import Optional

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set! Run 'export AIPROXY_TOKEN=your-token' before starting FastAPI.")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set! Run 'export GITHUB_TOKEN=your-github-token'.")

AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
GITHUB_API_URL = "https://api.github.com"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerResponse(BaseModel):
    answer: str

def get_github_repo_with_action():
    """Fetch the most recent repository where an action was triggered."""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # Step 1: Get the authenticated user's repositories
    user_resp = httpx.get(f"{GITHUB_API_URL}/user", headers=headers)
    if user_resp.status_code != 200:
        return "Error fetching user info."
    
    username = user_resp.json()["login"]
    
    repos_resp = httpx.get(f"{GITHUB_API_URL}/users/{username}/repos", headers=headers)
    if repos_resp.status_code != 200:
        return "Error fetching repositories."

    repos = repos_resp.json()

    # Step 2: Check each repo for recent GitHub Actions runs
    for repo in repos:
        repo_name = repo["name"]
        actions_resp = httpx.get(f"{GITHUB_API_URL}/repos/{username}/{repo_name}/actions/runs", headers=headers)

        if actions_resp.status_code == 200 and actions_resp.json()["total_count"] > 0:
            return f"https://github.com/{username}/{repo_name}"

    return "No recent GitHub Actions found."

def get_llm_answer(question: str, context: str = "") -> str:
    """Queries GPT-4o-mini through AIPROXY using httpx with optional PDF context."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    prompt = question if not context else f"Context: {context}\nQuestion: {question}"
    
    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a computational expert who outputs correct answers after interpreting and analysing the problem step by step. Use provided context strictly. Do not assume missing data. Return only the final answer, no explanations. Respond ONLY with the exact answer, no additional text, explanations or context. If the nswer is a number, only return the number. If the answer is a text, return only the precise text required."},
            {"role": "user", "content": prompt}
        ],
    }
    
    try:
        response = httpx.post(AIPROXY_BASE_URL, json=payload, headers=headers, timeout=60.0)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            return "AIPROXY monthly request limit reached. Try again next month!"
        elif response.status_code == 403:
            return "AIPROXY cost limit exceeded ($0.5). Requests are blocked!"
        else:
            return f"AIPROXY Error {response.status_code}: {response.text}"
    
    except httpx.TimeoutException:
        return "AIPROXY request timed out. Try again later or simplify your question."
    except httpx.RequestError as e:
        return f"Failed to connect to AIPROXY: {str(e)}"

@app.post("/api/", response_model=AnswerResponse)
async def solve_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """API endpoint to answer graded assignment questions with optional PDF context."""
    answer = get_llm_answer(question)
    return AnswerResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)