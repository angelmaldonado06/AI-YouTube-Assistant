from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Optional, Tuple
from prompts import create_queries_prompt
import json
from llms import get_llm
from langchain_core.output_parsers import StrOutputParser
from transcript import (
    format_transcript_entries,
    get_transcript,
    get_video_id,
    normalize_transcript_entries,
)
from helpers import is_cached, save_to_cache, load_from_cache

_reranker_cache = None

# MAIN ENTRY POINT: Orchestrates the full ingestion + indexing pipeline
def prepare_video(video_url) -> tuple[str, FAISS | None]:
    """Fetch video transcript and build FAISS index, using a disk cache when available."""
    if not video_url:
        return "", None

    video_id = get_video_id(video_url)
    if not video_id:
        return "", None

    embedding_model = create_embedding_model()

    if is_cached(video_id):
        return load_from_cache(video_id, embedding_model)

    fetched_transcript = get_transcript(video_url)
    if not fetched_transcript:
        return "", None

    transcript_entries = normalize_transcript_entries(fetched_transcript)
    processed_transcript = format_transcript_entries(transcript_entries)
    transcript_documents = build_transcript_documents(transcript_entries)
    faiss_index = create_faiss_index_from_documents(
        transcript_documents,
        embedding_model,
    )

    save_to_cache(video_id, processed_transcript, faiss_index)

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
def retrieve_documents(query, faiss_index, k=10) -> list:
    """Retrieve top-k most similar documents from FAISS index."""
    return faiss_index.similarity_search(query, k=k)


def generate_rephrased_queries(query: str) -> List:
    """Generate 3 alternative phrasings of the question for multi-query retrieval."""
    llm = get_llm()
    prompt = create_queries_prompt()
    chain = prompt | llm| StrOutputParser()

    response = chain.invoke({"query": query})

    print(f"MULTI-QUERY")
    print(f"{'='*70}")
    print(f"LLM Response: {response}")

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


def retrieve_context(query: str, faiss_index, time_range: Optional[dict] = None) -> Tuple[str, List[Document]]:
    """
    Retrieve and process documents for a query.
    Handles: multi-query expansion, deduplication, time filtering, context formatting.
    Returns: (formatted_context, retrieved_documents)
    """
    # Multi-query expansion
    queries = [query]
    rephrased = generate_rephrased_queries(query)
    queries.extend(rephrased)

    # Search FAISS with each query
    all_docs = []
    for q in queries:
        docs = retrieve_documents(q, faiss_index)
        all_docs.extend(docs)

    # deduplicate
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    #time range filtering
    if time_range:
        start = time_range['start_seconds']
        end = time_range['end_seconds']

        print(f"\nFILTERING BY TIME RANGE: {start} - {end} seconds")
        print(f"Docs before filter: {len(unique_docs)}")

        for doc in unique_docs[:3]:
            print(f"  Doc start: {doc.metadata.get('start_seconds')} - {doc.page_content[:60]}...")

        unique_docs = [doc for doc in unique_docs if start <= doc.metadata['start_seconds'] <= end]
        print(f"Docs after filter: {len(unique_docs)}")
    else:
        print("Time range None")

    # Format context
    context = "\n\n".join([f"{doc.page_content} (Timestamp: {doc.metadata.get('timestamp', 'N/A')})" for doc in unique_docs])

    return context, unique_docs


def get_reranker() -> CrossEncoder:
    """Create or return cached CrossEncoder model for reranking."""
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_cache