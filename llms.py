from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_generator_llm = None
_judge_llm = None

load_dotenv()

def get_llm() -> ChatOpenAI:
    """Create or return cached LLM instance."""
    global _generator_llm
    if _generator_llm is None:
        _generator_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)
    return _generator_llm

def get_eval_llm():
    """Evaluation LLM — Mistral Nemo as judge."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    return _judge_llm