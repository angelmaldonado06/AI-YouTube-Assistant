from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List
from prompts import create_queries_prompt
import json
from transcript import (
    format_transcript_entries,
    get_transcript,
    normalize_transcript_entries,
)

_llm_cache = None
_reranker_cache = None

# MAIN ENTRY POINT: Orchestrates the full ingestion + indexing pipeline
def prepare_video(video_url) -> tuple[str, FAISS | None]:
    """Fetch video transcript and build FAISS index."""
    if not video_url:
        return "", None

    fetched_transcript = get_transcript(video_url)
    if not fetched_transcript:
        return "", None

    transcript_entries = normalize_transcript_entries(fetched_transcript)
    processed_transcript = format_transcript_entries(transcript_entries)
    transcript_documents = build_transcript_documents(transcript_entries)
    embedding_model = create_embedding_model()
    faiss_index = create_faiss_index_from_documents(
        transcript_documents,
        embedding_model,
    )

    return processed_transcript, faiss_index

# INDEXING: Chunk, embed, and store documents
def build_transcript_documents(transcript_entries, chunk_size=800, chunk_overlap=100) -> list:
    """Split transcript entries into Document chunks with timestamps."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = [
        Document(
            page_content=f"Text: {entry['text']}",
            metadata={
                "timestamp": entry["timestamp"],
                "start_seconds": entry["start_seconds"],
            },
        )
        for entry in transcript_entries
    ]

    return text_splitter.split_documents(documents)


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Create HuggingFace embedding model for text vectorization."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def create_faiss_index_from_documents(documents, embedding_model) -> FAISS:
    """Build FAISS vector index from document embeddings."""
    return FAISS.from_documents(documents, embedding_model)


# RETRIEVAL: Search and re-phrase queries for better recall
def retrieve_documents(query, faiss_index, k=4) -> list:
    """Retrieve top-k most similar documents from FAISS index."""
    return faiss_index.similarity_search(query, k=k)


def retrieve(query, faiss_index, k=4) -> str:
    """Retrieve and format top-k documents as concatenated context string."""
    docs = retrieve_documents(query, faiss_index, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


def generate_rephrased_queries(question: str) -> List:
    """Generate 3 alternative phrasings of the question for multi-query retrieval."""
    llm = create_llm()
    prompt = create_queries_prompt()
    chain = prompt | llm

    response = chain.invoke({"question": question})

    print(f"\n{'='*70}")
    print(f"MULTI-QUERY")
    print(f"{'='*70}")
    print(f"Raw LLM Response: {response}")

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            queries = parsed.get("queries", [])
        else:
            queries = []

        return queries

    except json.JSONDecodeError as e:
        print(f"Error parsing queries: {e}")
        return []


# MODEL INITIALIZATION: Cached LLM and reranker instances
def create_llm() -> OllamaLLM:
    """Create or return cached Ollama LLM instance."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = OllamaLLM(model="llama3.1", temperature=0.3)
    return _llm_cache


def get_reranker() -> CrossEncoder:
    """Create or return cached CrossEncoder model for reranking."""
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_cache