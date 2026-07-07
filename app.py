import gradio as gr
from rag_pipeline import prepare_video
from llms import get_llm
from prompts import create_summary_prompt
from graph import create_initial_state, rag_graph

processed_transcript = ""
faiss_index = None
conversation_history = []
current_video_url = None

NON_VIDEO_MESSAGES = {
    "bye",
    "goodbye",
    "hi",
    "hello",
    "hey",
    "how are you",
    "what is your name",
    "what's your name",
    "thanks",
    "thank you",
    "what's up",
    "whats up",
    "who are you",
}


def is_non_video_message(question: str) -> bool:
    """return True when the message should not enter the RAG graph."""
    normalized = question.lower().strip(" .!?")
    return normalized in NON_VIDEO_MESSAGES


def answer_without_retrieval(question: str) -> str:
    """Handle simple messages that are not questions about the video."""
    normalized = question.lower().strip(" .!?")
    if not normalized:
        return "Ask me a question about the video."
    if normalized in {"what is your name", "what's your name", "who are you"}:
        return "I don’t have a personal name, but I’m your Personal AI YouTube Assistant. I help summarize videos and answer questions using the video transcript."
    if normalized in {"bye", "goodbye"}:
        return "Goodbye! Come back anytime with another question about the video."

    return "Hi! Ask me a question about the video, and I’ll answer using the video."

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
        summary_chain = summary_prompt | llm

        summary = summary_chain.invoke({
            "transcript" : processed_transcript
        })
        return summary
    else:
        return "No transcripts available"
    

def answer_question(video_url, question, from_min=None, to_min=None) -> str:
    """Answer a question based on video transcript using the RAG graph."""
    global processed_transcript, faiss_index, conversation_history, current_video_url

    question = question or ""
    if not question.strip() or is_non_video_message(question):
        return answer_without_retrieval(question)

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


with gr.Blocks() as interface:
    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="Enter YouTube URL"
    )

    summary_output = gr.Textbox(
        label="Video Summary",
        lines=6
    )

    summarize_btn = gr.Button("Summarize Video")

    summarize_btn.click(
        summarize_video,
        inputs=[video_url],
        outputs=[summary_output]
    )

    question_input = gr.Textbox(
        label="Ask a Question"
    )

    with gr.Row():
        from_min = gr.Number(label="From (min)", minimum=0, value=None, precision=1)
        to_min = gr.Number(label="To (min)", minimum=0, value=None, precision=1)

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    question_btn = gr.Button("Ask Question")

    question_btn.click(
        answer_question,
        inputs=[video_url, question_input, from_min, to_min],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    interface.launch()