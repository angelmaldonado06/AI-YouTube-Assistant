import gradio as gr
from rag_pipeline import prepare_video
from llms import get_llm
from prompts import (
    create_summary_prompt
)
from graph import rag_graph, RAGState

processed_transcript = ""
faiss_index = None
conversation_history = []

def summarize_video(video_url) -> str:
    """Fetch video transcript and generate a concise summary."""
    global processed_transcript, faiss_index

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
    

def answer_question(video_url, question) -> str:
    """Answer a question based on video transcript using the RAG graph."""
    global processed_transcript, faiss_index, conversation_history

    if not processed_transcript:
        processed_transcript, faiss_index = prepare_video(video_url)
    
    if processed_transcript and question:
        state = RAGState(
            query = question,
            video_url = video_url,
            processed_transcripts = processed_transcript,
            chunks = [],
            faiss_index = faiss_index,
            final_answer = "",
            retrieved_context = "",
            needs_retrieval = False,
            router_confidence = 0.0,
            routing_decision = "",
            retrieved_documents = [],
            conversation_history = conversation_history,
            attempt_count= 0,
            eval_score= 0.0,
            reflection_feedback = "" 
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

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    question_btn = gr.Button("Ask Question")

    question_btn.click(
        answer_question,
        inputs=[video_url, question_input],
        outputs=[answer_output]
    )

interface.launch()