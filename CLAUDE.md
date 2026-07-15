# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG-powered app that answers questions about a YouTube video from its transcript. FastAPI backend (`backend/`) + plain HTML/CSS/JS frontend (`frontend/`). Retrieval uses FAISS + HuggingFace embeddings; generation uses OpenAI `gpt-4o-mini` via LangChain; the question-answering flow is a LangGraph state machine with routing, multi-query retrieval, reranking, and a self-critique/revision loop.

## Commands

```bash
# Install dependencies (run from backend/)
cd backend
pip install -r requirements.txt

# Run the API server
cd backend
uvicorn main:app --reload

# Then open frontend/index.html directly in a browser
```

There is no test suite, linter, or build step configured in this repo.

### Evaluation (RAGAS)

```bash
cd backend
python evaluation.py --video-url "YOUTUBE_URL" --dataset sample_eval_dataset.json [--k 4]
```

Evaluates the pipeline with `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. The eval dataset is a JSON list of `{"question": ..., "ground_truth": ...}` objects (see `sample_eval_dataset.json`).

### Environment

Requires an `OPENAI_API_KEY` in a `.env` file at the repo root (used by both `backend/llms.py` generation/judge models).

## Architecture

### Request flow

`frontend/script.js` calls two endpoints exposed by `backend/main.py`:
- `POST /load` — `video_session.summarize_video()` → fetches/caches transcript, builds FAISS index, generates a summary.
- `POST /ask` — `video_session.answer_question()` → runs the LangGraph RAG workflow (`graph.py`) for a question, optionally scoped to a `from_min`/`to_min` timestamp window.
- `POST /clear` — resets in-memory conversation history.

`video_session.py` holds **module-level global state** (`processed_transcript`, `faiss_index`, `conversation_history`, `current_video_url`) — this is a single-session app, not multi-user/multi-tenant. Switching `video_url` resets that state.

### Ingestion pipeline (`rag_pipeline.py` + `transcript.py` + `cache.py`)

`prepare_video(video_url)` is the entry point:
1. `transcript.get_video_id()` extracts the video ID; `transcript.get_transcript()` fetches captions via `youtube_transcript_api`, preferring manually-created English transcripts over auto-generated ones.
2. If a disk cache exists for the video ID (`backend/cache/<video_id>/`, containing `faiss_index/` and `transcript.txt`), it's loaded instead of re-fetching/re-embedding (`cache.py`).
3. Otherwise transcript entries are normalized, chunked with `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) into `Document`s carrying `timestamp`/`start_seconds` metadata, embedded with `BAAI/bge-base-en-v1.5` via `HuggingFaceEmbeddings`, and indexed into FAISS. Results are saved to the disk cache.

### RAG workflow (`graph.py`)

A LangGraph `StateGraph` over `RAGState`, wired as:

```
START → router ─(generate)──────────────────→ generate → reflection ─(regenerate)─→ generate (loop, max 3 attempts)
              └(retrieve)→ retrieve ─(no_context)→ no_context ─────────────────────→ output
                                └(has_context)→ rerank → generate → reflection ─(output)─→ output → END
```

- **router_node**: LLM classifies whether the question needs transcript retrieval or can be answered from conversation alone (JSON output with `needs_transcript`/`confidence`; confidence < 0.6 or a parse failure falls back to `retrieve` for safety).
- **retrieve_node** (`rag_pipeline.retrieve_context`): multi-query expansion (3 LLM-generated rephrasings + original), FAISS similarity search per query, dedup by content, optional time-range filtering by `start_seconds`.
- **rerank_node**: cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, lazily cached) re-scores retrieved docs and keeps the top 4.
- **generation_node**: picks a prompt from `prompts.py` based on state — chat-only, context-grounded answer, or revision (when there's prior critique feedback).
- **reflection_node**: a separate judge LLM (`llms.get_eval_llm`, temperature 0) scores the answer and returns `decision`/`feedback` as JSON; a parse failure defaults to a passing score rather than looping forever.
- **should_regenerate**: loops back to `generate` while `eval_score < 7` or the judge says needs-improvement/fail, capped at 3 attempts, otherwise proceeds to `output`.
- **output_node**: appends the turn to `conversation_history` and returns `final_answer`.

All prompt templates live in `prompts.py` (summary, answer, revision, eval/judge, router, multi-query, chat).

### LLM instances (`llms.py`)

Two module-level cached `ChatOpenAI` instances: `get_llm()` (temperature 0.5, used for generation/routing/query rephrasing) and `get_eval_llm()` (temperature 0, used for judging/reflection and in `evaluation.py`).
