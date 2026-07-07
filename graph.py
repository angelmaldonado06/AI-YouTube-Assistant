from typing import TypedDict, Optional, List
import json
import logging
from rag_pipeline import retrieve_context, get_reranker
from prompts import create_answer_prompt, create_revision_prompt, create_eval_prompt
from llms import get_llm, get_eval_llm
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """
    Represents the state of a RAG (Retrieval-Augmented Generation) process.
    """
    query: str
    faiss_index: Optional[object]
    final_answer: str
    retrieved_context: str
    retrieved_documents: List[Document]
    conversation_history: List[BaseMessage]
    attempt_count: int
    eval_score: float
    reflection_feedback: str
    reflection_decision: str
    time_range: Optional[dict]

# TODO: Upgrade 7 — refactor this to a class-based Router with Pydantic BaseModel
# and tool binding. For now, keep it simple for Upgrade 1.

def create_initial_state(query: str, faiss_index, conversation_history: List[BaseMessage], time_range: Optional[dict] = None) -> RAGState:
    """Build the graph state in one place so app.py stays simple."""
    return RAGState(
        query=query,
        faiss_index=faiss_index,
        final_answer="",
        retrieved_context="",
        retrieved_documents=[],
        conversation_history=conversation_history,
        attempt_count=0,
        eval_score=0.0,
        reflection_feedback="",
        reflection_decision="",
        time_range=time_range
    )

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant transcript chunks based on query."""
    context, documents = retrieve_context(
        query=state["query"],
        faiss_index=state["faiss_index"],
        time_range=state.get("time_range")
    )

    state['retrieved_context'] = context
    state["retrieved_documents"] = documents

    return state

def rerank_node(state: RAGState) ->RAGState:
    '''Re-rank retrieved documents by relevance using cross-encoder.''' 
    documents = state["retrieved_documents"]
    question = state["query"]
    ranker = get_reranker()

    pairs = [(question, doc.page_content) for doc in documents]
    scores = ranker.predict(pairs)

    ranked_docs = [doc for score, doc in sorted(zip(scores, documents), key=lambda x: -x[0])][:4]

    logger.debug(f"\n{'='*70}")
    logger.debug(f"RERANKING RESULTS")
    logger.debug(f"{'='*70}")
    for i, (score, doc) in enumerate(sorted(zip(scores, documents), key=lambda x: -x[0])[:4], 1):
        logger.debug(f"Rank {i} (score: {score:.4f})")
        logger.debug(f"  {doc.page_content[:100]}...")
    logger.debug(f"{'='*70}\n")
    
    context = "\n\n".join([f"{doc.page_content} (Timestamp: {doc.metadata.get('timestamp', 'N/A')})" for doc in ranked_docs])

    state["retrieved_documents"] = ranked_docs
    state["retrieved_context"] = context

    return state

def generation_node(state: RAGState) -> RAGState:
    """Generate or revise answer using LLM with retrieved context."""
    llm = get_llm()

    if state['attempt_count'] > 0 and state.get('reflection_feedback'):
        prompt = create_revision_prompt()
        inputs = {
            "context": state['retrieved_context'],
            "question": state['query'],
            "previous_answer": state['final_answer'],
            "feedback": state['reflection_feedback'],
        }
    else:
        prompt = create_answer_prompt()
        inputs = {
            "context": state['retrieved_context'],
            "question": state['query'],
            "conversation_history": state['conversation_history'],
        }

    answer = (prompt | llm).invoke(inputs)
    state['final_answer'] = answer

    return state

def critique_node(state:RAGState) -> RAGState:
    '''Evaluate output using external evaluator and decide if it needs revision'''
    evaluator = get_eval_llm()
    prompt = create_eval_prompt()
    chain = prompt | evaluator

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

        logger.debug(f"  Score: {score}")
        logger.debug(f"  Decision: {decision}")
        logger.debug(f"  Feedback: {feedback}")

        state['reflection_feedback'] = feedback
        state['reflection_decision'] = decision
        state['eval_score'] = score
        state['attempt_count'] += 1

    except json.JSONDecodeError as e:
        logger.debug(f"JSON Parse Error: {e}")
        logger.debug("Evaluation failed. Keeping current answer instead of looping forever.")
        state['reflection_feedback'] = "Evaluator returned invalid JSON."
        state['reflection_decision'] = "PASS"
        state['eval_score'] = 7
        state['attempt_count'] += 1
        logger.debug(f"{'='*70}\n")
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
graph_builder.add_node("critique", critique_node)
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
graph_builder.add_conditional_edges("critique", should_regenerate, {
    "regenerate" : "generate",
    "output":"output"
})

graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", "critique")
graph_builder.add_edge("output", END)

# Compile the graph
rag_graph = graph_builder.compile()
