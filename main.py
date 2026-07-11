from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from app import summarize_video, answer_question, clear_conversation

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoadRequest(BaseModel):
    video_url:str

@app.post('/load')
def load(body: LoadRequest):
    summary = summarize_video(body.video_url)
    return {"summary":summary}

class AskRequest(BaseModel):
    video_url:str
    question:str
    from_min: Optional[float] = None
    to_min:Optional[float] = None

@app.post('/ask')
def ask(body: AskRequest):
    answer = answer_question(body.video_url, body.question, body.from_min, body.to_min)
    return {"answer":answer}

@app.post('/clear')
def clear():
    clear_conversation()
    return {"status": "cleared"}
