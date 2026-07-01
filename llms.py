from langchain_ollama import OllamaLLM

_llm_cache = None
_judge_cache = None


def get_llm() -> OllamaLLM:
    """Create or return cached Ollama LLM instance."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = OllamaLLM(model="llama3.1", temperature=0.3)
    return _llm_cache

def get_eval_llm():
    """Evaluation LLM — Mistral Nemo as judge."""
    global _judge_cache
    if _judge_cache is None:
        _judge_cache = OllamaLLM(
            model="mistral-nemo",
            temperature=0
        )
    return _judge_cache