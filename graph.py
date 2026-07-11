from typing import TypedDict, Optional, List
import json
from rag_pipeline import retrieve_context, get_reranker
from prompts import create_answer_prompt, create_revision_prompt, create_eval_prompt, create_router_prompt
from llms import get_llm, get_eval_llm
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


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
    critique_feedback: str
    critique_decision: str
    time_range: Optional[dict]
    routing_decision: str

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
        critique_feedback="",
        critique_decision="",
        time_range=time_range
    )

def router_node(state: RAGState) -> dict:
    """Decide whether to retrieve from transcript or answer directly."""
    llm = get_llm()
    router_prompt = create_router_prompt()
    router_chain = router_prompt | llm | StrOutputParser()
    response = router_chain.invoke({"question": state['query']})

    print(f"\n{'='*70}")
    print(f"Query: {state['query']}")
    print(f"{'='*70}")

    try:
        parsed = json.loads(response)
        print(f"Parsed JSON: {parsed}")

        needs_transcript = parsed.get("needs_transcript", False)
        confidence = parsed.get("confidence", 0.0)

        print(f"  needs_transcript: {needs_transcript}")
        print(f"  confidence: {confidence}")

        # Fallback: if confidence < 0.6, always retrieve
        if confidence < 0.6:
            decision = "retrieve"
            print(f"\nLow confidence ({confidence}) → FALLBACK to retrieve")
        else:
            decision = "retrieve" if needs_transcript else "generate"
            print(f"\nRouting decision: {decision}")

        print(f"{'='*70}")
        return {"routing_decision": decision}

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Defaulting to 'retrieve' for safety")
        print(f"{'='*70}\n")
        return {"routing_decision": "retrieve"}


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
    if not documents:
        state["retrieved_context"] = ""
        return state

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

    state["retrieved_documents"] = ranked_docs
    state["retrieved_context"] = context

    return state

def generation_node(state: RAGState) -> RAGState:
    """Generate or revise answer using LLM with retrieved context."""
    llm = get_llm()

    if state['attempt_count'] > 0 and state.get('critique_feedback'):
        prompt = create_revision_prompt()
        inputs = {
            "context": state['retrieved_context'],
            "question": state['query'],
            "previous_answer": state['final_answer'],
            "feedback": state['critique_feedback'],
        }
    else:
        prompt = create_answer_prompt()
        inputs = {
            "context": state['retrieved_context'],
            "question": state['query'],
            "conversation_history": state['conversation_history'],
        }

    answer = (prompt | llm | StrOutputParser()).invoke(inputs)

    state['final_answer'] = answer

    return state

def no_context_node(state: RAGState) -> RAGState:
    """Return a clear answer when retrieval found no usable context."""
    state["final_answer"] = "I could not find relevant transcript context for that question."
    return state

def reflection_node(state:RAGState) -> RAGState:
    '''Evaluate output using external evaluator and decide if it needs revision'''
    evaluator = get_eval_llm()
    prompt = create_eval_prompt()
    chain = prompt | evaluator | StrOutputParser()

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

        state['critique_feedback'] = feedback
        state['critique_decision'] = decision
        state['eval_score'] = score
        state['attempt_count'] += 1

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print("Evaluation failed. Keeping current answer instead of looping forever.")
        state['critique_feedback'] = "Evaluator returned invalid JSON."
        state['critique_decision'] = "PASS"
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
graph_builder.add_node("router", router_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("generate", generation_node)
graph_builder.add_node("no_context", no_context_node)
graph_builder.add_node("reflection", reflection_node)
graph_builder.add_node("output", output_node)

# Set entry point: route first, then decide whether to retrieve
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", lambda state: state["routing_decision"], {
    "retrieve": "retrieve",
    "generate": "generate",
})

def should_regenerate(state):
    critique_decision = state["critique_decision"].lower()
    needs_regen = (
        state["eval_score"] < 7
        or critique_decision in {"needs_improvement", "fail"}
    )

    if needs_regen and state['attempt_count'] < 3:
        return "regenerate"
    else:
        return "output"
graph_builder.add_conditional_edges("reflection", should_regenerate, {
    "regenerate" : "generate",
    "output":"output"
})

def has_context(state):
    return "has_context" if state["retrieved_context"].strip() else "no_context"

graph_builder.add_conditional_edges("retrieve", has_context, {
    "has_context": "rerank",
    "no_context": "no_context",
})
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", "reflection")
graph_builder.add_edge("no_context", "output")
graph_builder.add_edge("output", END)

# Compile the graph
rag_graph = graph_builder.compile()
