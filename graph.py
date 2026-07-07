from typing import TypedDict, Optional, List
import json
from rag_pipeline import generate_rephrased_queries, retrieve_documents, get_reranker
from prompts import create_qa_prompt, create_eval_prompt
from llms import get_llm, get_eval_llm
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class RAGState(TypedDict):
    """
    Represents the state of a RAG (Retrieval-Augmented Generation) process.
    """
    query: str
    video_url: str
    processed_transcript: str
    chunks: List[dict]
    faiss_index: Optional[object]
    final_answer: str
    retrieved_context: str
    has_retrieved_context: bool
    retrieved_documents: List[Document]
    conversation_history: List[BaseMessage]
    attempt_count: int
    eval_score: float
    reflection_feedback: str 
    reflection_decision: str
    time_range: Optional[dict]
    has_empty_context: bool

# TODO: Upgrade 7 — refactor this to a class-based Router with Pydantic BaseModel
# and tool binding. For now, keep it simple for Upgrade 1.

def create_initial_state(query: str, video_url: str, processed_transcript: str, faiss_index, conversation_history: List[BaseMessage], time_range: Optional[dict] = None) -> RAGState:
    """Build the graph state in one place so app.py stays simple."""
    return RAGState(
        query=query,
        video_url=video_url,
        processed_transcript=processed_transcript,
        chunks=[],
        faiss_index=faiss_index,
        final_answer="",
        retrieved_context="",
        has_retrieved_context=False,
        retrieved_documents=[],
        conversation_history=conversation_history,
        attempt_count=0,
        eval_score=0.0,
        reflection_feedback="",
        reflection_decision="",
        time_range=time_range,
        has_empty_context=False
    )

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant transcript chunks based on query."""

    #generating multi-queries
    queries = [state["query"]]
    rephrased = generate_rephrased_queries(state['query'])
    queries.extend(rephrased)

    #search faiss
    all_docs = []
    for q in queries:
        docs = retrieve_documents(q, state["faiss_index"])
        all_docs.extend(docs)


    #remove duplicates
    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    if state.get('time_range'):
        start = state['time_range']['start_seconds']
        end = state['time_range']['end_seconds']

        print(f"\nFILTERING BY TIME RANGE: {start} - {end} seconds")
        print(f"Docs before filter: {len(unique_docs)}")

        for doc in unique_docs[:3]:
            print(f"  Doc start: {doc.metadata.get('start_seconds')} - {doc.page_content[:60]}...")

        unique_docs = [doc for doc in unique_docs if start <= doc.metadata['start_seconds'] <= end]
        print(f"Docs after filter: {len(unique_docs)}")
    else:
        print("Time range None")

    context = "\n\n".join([f"{doc.page_content} (Timestamp: {doc.metadata.get('timestamp', 'N/A')})" for doc in unique_docs])

    state['retrieved_context'] = context
    state["retrieved_documents"] = unique_docs
    state['has_retrieved_context'] = bool(context.strip())
    state['has_empty_context'] = not bool(context.strip())

    return state

def rerank_node(state: RAGState) ->RAGState:
    '''Re-rank retrieved documents by relevance using cross-encoder.''' 
    documents = state["retrieved_documents"]
    question = state["query"]
    ranker = get_reranker()

    pairs = [(question, doc.page_content) for doc in documents]
    scores = ranker.predict(pairs)

    ranked_docs = [doc for score, doc in sorted(zip(scores, documents), key=lambda x: -x[0])][:4]

    print(f"\n{'='*70}")
    print(f"RERANKING RESULTS")
    print(f"{'='*70}")
    for i, (score, doc) in enumerate(sorted(zip(scores, documents), key=lambda x: -x[0])[:4], 1):
        print(f"Rank {i} (score: {score:.4f})")
        print(f"  {doc.page_content[:100]}...")
    print(f"{'='*70}\n")
    
    context = "\n\n".join([f"{doc.page_content} (Timestamp: {doc.metadata.get('timestamp', 'N/A')})" for doc in ranked_docs])

    state["retrieved_context"] = context

    return state

def generation_node(state: RAGState) -> RAGState:
    """Generate answer using LLM with retrieved context."""
    llm = get_llm()

    prompt = create_qa_prompt()
    chain = prompt | llm

    if state['attempt_count'] > 0 and state.get('reflection_feedback'):
        feedback_guidance = f"""REVISION TASK:
        Your previous answer: {state['final_answer']}
        Evaluator feedback: {state['reflection_feedback']}
        Please revise your answer to address the feedback above. Keep what was good, fix what was lacking.
        """
    else:
        feedback_guidance = ""

    answer = chain.invoke({
        "context": state['retrieved_context'],
        "question": state['query'],
        'conversation_history': state['conversation_history'],
        'feedback_guidance': feedback_guidance
    })

    state['final_answer'] = answer

    return state

def reflection_node(state:RAGState) -> RAGState:
    '''Critique output and revise it'''
    judge = get_eval_llm()
    prompt = create_eval_prompt()
    chain = prompt | judge

    response = chain.invoke({
        'context': state["retrieved_context"],
        'question': state["query"],
        'answer': state["final_answer"],
    })

    try:
        parsed = json.loads(response)

        score = parsed.get("score", 0)
        decision = parsed.get("decision", "Fail")
        feedback = parsed.get("feedback", "")

        print(f"  Score: {score}")
        print(f"  Decision: {decision}")
        print(f"  Feedback: {feedback}")

        state['reflection_feedback'] = feedback
        state['reflection_decision'] = decision
        state['eval_score'] = score
        state['attempt_count'] += 1

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print("Evaluation failed. Keeping current answer instead of looping forever.")
        state['reflection_feedback'] = "Evaluator returned invalid JSON."
        state['reflection_decision'] = "PASS"
        state['eval_score'] = 7
        state['attempt_count'] += 1
        print(f"{'='*70}\n")
    return state

def output_node(state: RAGState) -> RAGState:
    """Return final answer from state."""
    conversation_history = state["conversation_history"] + [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["final_answer"]),
    ]
    return {
        "final_answer": state["final_answer"],
        "conversation_history": conversation_history,
    }

graph_builder = StateGraph(RAGState)

# Add all nodes
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("generate", generation_node)
graph_builder.add_node("reflection", reflection_node)
graph_builder.add_node("output", output_node)

# Set entry point (always retrieve for YouTube assistant)
graph_builder.add_edge(START, "retrieve")

def should_regenerate(state):
    reflection_decision = state["reflection_decision"].lower()
    needs_regen = (
        state["eval_score"] < 7
        or reflection_decision in {"needs_improvement", "fail"}
    )

    if needs_regen and state['attempt_count'] < 3:
        return "regenerate"
    else:
        return "output"
graph_builder.add_conditional_edges("reflection", should_regenerate, {
    "regenerate" : "generate",
    "output":"output"
})

graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", "reflection")
graph_builder.add_edge("output", END)

# Compile the graph
rag_graph = graph_builder.compile()
