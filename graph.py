from typing import TypedDict, Optional, List
import json
from rag_pipeline import create_llm, retrieve
from prompts import create_qa_prompt, create_general_prompt, create_router_prompt
from langgraph.graph import StateGraph, END, START

class RAGState(TypedDict):
    """
    Represents the state of a RAG (Retrieval-Augmented Generation) process.
    """
    query: str
    video_url: str
    processed_transcripts: str
    chunks: List[dict]
    faiss_index: Optional[object]
    final_answer: str
    retrieved_context: str
    needs_retrieval: bool
    router_confidence: float
    routing_decision: str


# TODO: Upgrade 7 — refactor this to a class-based Router with Pydantic BaseModel
# and tool binding. For now, keep it simple for Upgrade 1.
    

def router_node(state: RAGState) -> dict:
    """Decide whether to retrieve from transcript or answer directly."""
    llm = create_llm()
    router_prompt = create_router_prompt()
    router_chain = router_prompt | llm
    raw_response = router_chain.invoke({"question": state['query']})

    print(f"\n{'='*70}")
    print(f"ROUTER DEBUG | Query: {state['query']}")
    print(f"{'='*70}")
    print(f"Raw LLM Response: {raw_response}")

    try:
        parsed = json.loads(raw_response)
        print(f"Parsed JSON: {parsed}")

        needs_transcript = parsed.get("needs_transcript", False)
        confidence = parsed.get("confidence", 0.0)

        print(f"  needs_transcript: {needs_transcript}")
        print(f"  confidence: {confidence}")

        # Fallback: if confidence < 0.6, always retrieve
        if confidence < 0.6:
            decision = "retrieve"
            print(f"Low confidence ({confidence}) → FALLBACK to retrieve")
        else:
            decision = "retrieve" if needs_transcript else "generate"
            print(f"Routing decision: {decision}")

        print(f"{'='*70}\n")
        return {"routing_decision": decision}

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Defaulting to 'retrieve' for safety")
        print(f"{'='*70}\n")
        return {"routing_decision": "retrieve"}


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant transcript chunks based on query."""
    context = retrieve(state['query'], state['faiss_index'])
    state['retrieved_context'] = context
    needs_retrieval = bool(context.strip())
    state['needs_retrieval'] = needs_retrieval
    return state

def generation_node(state: RAGState) -> RAGState:
    """Generate answer using LLM with retrieved context."""
    llm = create_llm()

    prompt = create_general_prompt() if state['routing_decision'] == 'generate' else create_qa_prompt()
    chain = prompt | llm

    if state['routing_decision'] == 'generate':
        answer = chain.invoke({"question": state['query']})
    else:
        answer = chain.invoke({"context": state['retrieved_context'], "question": state['query']})
    
    state['final_answer'] = answer
    return state

def output_node(state: RAGState) -> RAGState:
    """Return final answer from state."""
    return {"final_answer": state['final_answer']}


graph_builder = StateGraph(RAGState)

# Add all nodes
graph_builder.add_node("router", router_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generation_node)
graph_builder.add_node("output", output_node)

# Set entry point
graph_builder.add_edge(START, "router")

def route_decision(state):
    return state.get("routing_decision", "generate")

graph_builder.add_conditional_edges("router", route_decision)
# Connect remaining edges
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "output")
graph_builder.add_edge("output", END)

# Compile the graph
rag_graph = graph_builder.compile()
