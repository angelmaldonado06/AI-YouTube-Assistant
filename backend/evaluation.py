import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from llms import get_eval_llm
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_core.output_parsers import StrOutputParser
from prompts import create_answer_prompt
from rag_pipeline import (
    create_embedding_model,
    prepare_video,
    retrieve_documents,
)


def load_eval_questions(dataset_path) -> list[dict]:
    """Load evaluation questions and ground truth from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    required_keys = {"question", "ground_truth"}
    for index, item in enumerate(payload, start=1):
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(
                f"Item {index} is missing required keys: {sorted(missing_keys)}"
            )

    return payload


def build_context(documents) -> str:
    """Join document page contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer(question, vectorstore, qa_chain, retrieval_k) -> tuple[str, list]:
    """Retrieve documents and generate answer from question."""
    retrieved_docs = retrieve_documents(question, vectorstore, k=retrieval_k)
    context = build_context(retrieved_docs)
    answer = qa_chain.invoke(
        {
            "context": context,
            "question": question,
            "conversation_history": [],
        }
    )

    return answer, retrieved_docs


def build_eval_rows(video_url, eval_questions, retrieval_k=4) -> list[dict]:
    """Generate evaluation rows with answers and contexts for all questions."""
    _, vectorstore = prepare_video(video_url)
    if vectorstore is None:
        raise ValueError("Could not prepare the video transcript for evaluation.")

    qa_llm = get_eval_llm()
    qa_prompt = create_answer_prompt()
    qa_chain = qa_prompt | qa_llm | StrOutputParser()

    rows = []

    for item in eval_questions:
        question = item["question"]
        answer, retrieved_docs = generate_answer(
            question=question,
            vectorstore=vectorstore,
            qa_chain=qa_chain,
            retrieval_k=retrieval_k,
        )
        contexts = [doc.page_content for doc in retrieved_docs]

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )

    return rows


def run_ragas_eval(video_url, dataset_path, retrieval_k=4):
    """Evaluate RAG system on video using RAGAS metrics."""
    eval_questions = load_eval_questions(dataset_path)
    rows = build_eval_rows(
        video_url=video_url,
        eval_questions=eval_questions,
        retrieval_k=retrieval_k,
    )

    dataset = Dataset.from_list(rows)
    eval_llm = get_eval_llm()
    embedding_model = create_embedding_model()

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=LangchainLLMWrapper(eval_llm),
        embeddings=LangchainEmbeddingsWrapper(embedding_model),
    )

    return result


def main() -> None:
    """Parse CLI arguments and run RAGAS evaluation."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the YouTube RAG app.")
    parser.add_argument("--video-url", required=True, help="Target YouTube video URL.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSON file containing question/ground_truth pairs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve for each evaluation question.",
    )
    args = parser.parse_args()

    result = run_ragas_eval(
        video_url=args.video_url,
        dataset_path=Path(args.dataset),
        retrieval_k=args.k,
    )

    print(result)


if __name__ == "__main__":
    main()
