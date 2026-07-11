from rag_pipeline import prepare_video
from llms import get_llm
from prompts import create_summary_prompt
from graph import create_initial_state, rag_graph
from langchain_core.output_parsers import StrOutputParser

processed_transcript = ""
faiss_index = None
conversation_history = []
current_video_url = None


def summarize_video(video_url) -> str:
    """Fetch video transcript and generate a concise summary."""
    global processed_transcript, faiss_index, conversation_history, current_video_url
    
    if current_video_url != video_url:
        processed_transcript = ""
        faiss_index = None
        conversation_history = []
        current_video_url = video_url

    processed_transcript, faiss_index = prepare_video(video_url)
    
    if processed_transcript:
        llm = get_llm()
        summary_prompt = create_summary_prompt()
        summary_chain = summary_prompt | llm | StrOutputParser()

        summary = summary_chain.invoke({
            "transcript" : processed_transcript
        })
        return summary
    else:
        return "No transcripts available"
    

def answer_question(video_url, question, from_min=None, to_min=None) -> str:
    """Answer a question based on video transcript using the RAG graph."""
    global processed_transcript, faiss_index, conversation_history, current_video_url

    time_range = None
    if from_min is not None or to_min is not None:
        if from_min is None or to_min is None:
            return "Please provide both a start and end time."
        if from_min < 0 or to_min <= 0 or from_min >= to_min:
            return "Please enter a valid time range where From is before To."
        time_range = {
            "start_seconds": int(from_min * 60),
            "end_seconds": int(to_min * 60)
        }

    if current_video_url != video_url:
        processed_transcript = ""
        faiss_index = None
        conversation_history = []
        current_video_url = video_url

    if not processed_transcript:
        processed_transcript, faiss_index = prepare_video(video_url)
    
    if processed_transcript and question:
        state = create_initial_state(
            query=question,
            faiss_index=faiss_index,
            conversation_history=conversation_history,
            time_range=time_range
        )

        result = rag_graph.invoke(state)
        conversation_history = result['conversation_history']
        return result['final_answer']
    else:
        return "No transcript available"


def clear_conversation() -> None:
    """Reset conversation memory for the current video."""
    global conversation_history
    conversation_history = []