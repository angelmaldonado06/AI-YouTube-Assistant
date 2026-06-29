from typing import TypedDict, Optional, List
import json
from rag_pipeline import create_llm, generate_rephrased_queries, retrieve_documents, get_reranker
from prompts import create_qa_prompt, create_general_prompt, create_router_prompt
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document


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
    retrieved_documents: List[Document]


# TODO: Upgrade 7 — refactor this to a class-based Router with Pydantic BaseModel
# and tool binding. For now, keep it simple for Upgrade 1.
    

def router_node(state: RAGState) -> dict:
    """Decide whether to retrieve from transcript or answer directly."""
    llm = create_llm()
    router_prompt = create_router_prompt()
    router_chain = router_prompt | llm
    raw_response = router_chain.invoke({"question": state['query']})

    print(f"\n{'='*70}")
    print(f"Query: {state['query']}")
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
    
    context = "\n\n".join([doc.page_content for doc in unique_docs])

    state['retrieved_context'] = context
    state["retrieved_documents"] = unique_docs
    state['needs_retrieval'] = bool(context.strip())

    return state

def rerank_node(state: RAGState) ->RAGState:
    '''Re-rank retrieved documents by relevance using cross-encoder.''' 
    documents = state["retrieved_documents"]
    question = state["query"]
    ranker = get_reranker()

    pairs = [(question, doc.page_content) for doc in documents]
    scores = ranker.predict(pairs)

    ranked_docs = [doc for score, doc in sorted(zip(scores, documents), key=lambda x: -x[0])][:3]

    print(f"\n{'='*70}")
    print(f"RERANKING RESULTS")
    print(f"{'='*70}")
    for i, (score, doc) in enumerate(sorted(zip(scores, documents), key=lambda x: -x[0])[:3], 1):
        print(f"Rank {i} (score: {score:.4f})")
        print(f"  {doc.page_content[:100]}...")
    print(f"{'='*70}\n")
    
    context = "\n\n".join([doc.page_content for doc in ranked_docs]) 

    state["retrieved_context"] = context

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
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("output", output_node)

# Set entry point
graph_builder.add_edge(START, "router")

def route_decision(state):
    return state.get("routing_decision", "generate")

graph_builder.add_conditional_edges("router", route_decision)
# Connect remaining edges
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", "output")
graph_builder.add_edge("output", END)

# Compile the graph
rag_graph = graph_builder.compile()
