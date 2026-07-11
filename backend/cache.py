import os
from langchain_community.vectorstores import FAISS

CACHE_DIR = "cache"

def get_cache_paths(video_id: str) -> tuple[str, str]:
    """Return (faiss_index_dir, transcript_file_path) for a given video ID."""
    video_folder = os.path.join(CACHE_DIR, video_id)
    faiss_index_dir = os.path.join(video_folder, "faiss_index")  
    transcript_path = os.path.join(video_folder, "transcript.txt")  
    return faiss_index_dir, transcript_path


def is_cached(video_id:str) -> bool:
    """Check whether a cached FAISS index and transcript exist for this video."""
    faiss_index_dir, transcript_path = get_cache_paths(video_id)
    return os.path.exists(faiss_index_dir) and os.path.exists(transcript_path)


def save_to_cache(video_id: str, processed_transcript: str, faiss_index: FAISS) -> None:
    """Persist the transcript text and FAISS index to disk for this video."""
    faiss_index_dir, transcript_path = get_cache_paths(video_id)

    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    faiss_index.save_local(faiss_index_dir)

    with open(transcript_path, "w") as f:
        f.write(processed_transcript)


def load_from_cache(video_id: str, embedding_model) -> tuple[str, FAISS]:
    """Load the cached transcript text and FAISS index for this video."""
    faiss_index_dir, transcript_path = get_cache_paths(video_id)

    faiss_index = FAISS.load_local(
        faiss_index_dir,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    with open(transcript_path, "r") as f:
        processed_transcript = f.read()

    return processed_transcript, faiss_index