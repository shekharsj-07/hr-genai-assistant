def get_llm():
    """
    Returns the best available LLM backend.
    Priority:
    1. Ollama (if installed and running)
    2. HuggingFace local model (pure Python fallback)
    """
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="mistral", temperature=0)
    except Exception:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline

        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )

        return HuggingFacePipeline(pipeline=hf_pipeline)