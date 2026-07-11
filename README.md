# AI YouTube Assistant

A RAG-powered app that watches YouTube videos for you — ask it questions and it answers from the actual transcript instead of guessing.

## Overview

This project retrieves a video's transcript, splits it into chunks, embeds those chunks into a vector store, and uses an LLM to generate summaries and grounded answers. It also includes a RAG evaluation step with RAGAS so retrieval quality and answer quality can be measured instead of assumed.

## Features

- Generate concise summaries of YouTube videos
- Ask questions about a video and get transcript-grounded answers
- Preserve timestamps in retrieved transcript chunks
- Run questions through a LangGraph RAG workflow
- Improve recall with multi-query retrieval
- Re-rank retrieved chunks before generation
- Use conversation memory for follow-up questions
- Critique generated answers and revise when needed
- Evaluate the RAG pipeline with RAGAS

## How It Works

### Pipeline

```text
          YouTube URL
              ↓
      Transcript Retrieval
              ↓
        Text Chunking
              ↓
         Embeddings
              ↓
      FAISS Vector Search
              ↓
   LangGraph Retrieval Workflow
              ↓
        LLM (GPT-4o-mini)
              ↓
    Summary / Question Answering
              ↓
      RAG Evaluation (RAGAS)
```

### Tech Stack

- LangChain for prompt orchestration and chaining
- LangGraph for the question-answering workflow, memory, and critique loop
- FAISS for vector search
- OpenAI's gpt-4o-mini for generation and evaluation
- Hugging Face embeddings for semantic retrieval
- FastAPI backend with a plain HTML/CSS/JS frontend
- RAGAS for evaluation

## Installation

1. Clone the repository.
2. Install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Add an `OPENAI_API_KEY` to a `.env` file at the repo root (generation and evaluation both use `gpt-4o-mini`).

## Run the App

```bash
cd backend
uvicorn main:app --reload
```

Then open `frontend/index.html` in your browser, paste a YouTube URL, and generate a summary or ask questions about the video.

## Evaluation With RAGAS

RAG systems should be evaluated, not judged only by whether the answers sound fluent. This project includes an evaluation script that runs a set of reference questions through the RAG pipeline and scores the results with RAGAS.

### Metrics Used

- `faithfulness`: checks whether the answer is supported by the retrieved context
- `answer_relevancy`: checks whether the answer addresses the question
- `context_precision`: checks whether the retrieved chunks are useful and not noisy
- `context_recall`: checks whether retrieval found enough relevant information

### Evaluation Dataset

Use `sample_eval_dataset.json` as a template. Each item should include:

- `question`: a realistic user question
- `ground_truth`: the reference answer expected from the transcript

Example:

```json
[
  {
    "question": "What is the main topic of the video?",
    "ground_truth": "The video explains..."
  }
]
```

### Run Evaluation

From inside `backend/`:

```bash
python evaluation.py --video-url "YOUTUBE_URL_HERE" --dataset sample_eval_dataset.json
```

## Project Structure

```text
youtube-assistant
|
|-- backend/
|   |-- main.py                  (FastAPI entrypoint)
|   |-- video_session.py         (session state + orchestration)
|   |-- cache.py                 (video disk cache)
|   |-- graph.py
|   |-- rag_pipeline.py
|   |-- transcript.py
|   |-- prompts.py
|   |-- llms.py
|   |-- evaluation.py
|   |-- requirements.txt
|   `-- sample_eval_dataset.json
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- style.css
`-- README.md
```