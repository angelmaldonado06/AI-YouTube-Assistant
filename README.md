# AI YouTube Research Assistant using RAG

AI-powered application that summarizes YouTube videos and answers questions about their content using Retrieval-Augmented Generation (RAG).

## Overview

This project retrieves a video's transcript, splits it into chunks, embeds those chunks into a vector store, and uses an LLM to generate summaries and grounded answers. It also includes a RAG evaluation step with RAGAS so retrieval quality and answer quality can be measured instead of assumed.

## Features

- Generate concise summaries of YouTube videos
- Ask questions about a video and get transcript-grounded answers
- Preserve timestamps in retrieved transcript chunks
- Route questions through a LangGraph RAG workflow
- Improve recall with multi-query retrieval
- Re-rank retrieved chunks before generation
- Use conversation memory for follow-up questions
- Reflect on generated answers and revise when needed
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
   LangGraph Router / Retrieval
              ↓
        LLM (Llama 3.1)
              ↓
    Summary / Question Answering
              ↓
      RAG Evaluation (RAGAS)
```

### Tech Stack

- LangChain for prompt orchestration and chaining
- LangGraph for routing, memory, and reflection flow
- FAISS for vector search
- Ollama for local model serving
- Llama 3.1 for generation
- Hugging Face embeddings for semantic retrieval
- Gradio for the user interface
- RAGAS for evaluation

## Installation

1. Clone the repository.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install Ollama from [ollama.com](https://ollama.com/).
4. Pull the model:

```bash
ollama pull llama3.1
```

## Run the App

```bash
python app.py
```

Then open the Gradio interface in your browser, paste a YouTube URL, and generate a summary or ask questions about the video.

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

```bash
python evaluation.py --video-url "YOUTUBE_URL_HERE" --dataset sample_eval_dataset.json
```

## Project Structure

```text
youtube-assistant
|
|-- app.py
|-- evaluation.py
|-- graph.py
|-- llms.py
|-- prompts.py
|-- rag_pipeline.py
|-- requirements.txt
|-- sample_eval_dataset.json
|-- transcript.py
`-- README.md
```

## Why This Project Matters

This project demonstrates the core parts of a practical RAG system:

- transcript ingestion
- chunking and vector retrieval
- grounded question answering
- local LLM inference
- evaluation of retrieval and generation quality

It is a good example of how to build a small but complete RAG application with both user-facing functionality and an evaluation loop.
