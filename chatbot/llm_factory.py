import requests
from typing import Tuple

# -----------------------------
# Ollama Configuration
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "mistral"

# -----------------------------
# HuggingFace Fallback
# -----------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

HF_MODEL_NAME = "distilgpt2"


# -----------------------------
# Ollama Availability Check
# -----------------------------
def ollama_available() -> bool:
    """
    Checks whether Ollama server is running locally.
    """
    try:
        requests.get(OLLAMA_TAGS_URL, timeout=2)
        return True
    except Exception:
        return False


# -----------------------------
# Ollama Generator
# -----------------------------
def ollama_generate(prompt: str) -> str:
    """
    Generate response using Ollama (Mistral).
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["response"]


# -----------------------------
# HuggingFace Generator (CausalLM)
# -----------------------------
# NOTE: Loaded lazily to avoid slow startup if Ollama exists
_hf_pipeline = None


def hf_generate(prompt: str) -> str:
    """
    Generate response using HuggingFace local model (distilgpt2).
    """
    global _hf_pipeline

    if _hf_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)

        _hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
        )

    output = _hf_pipeline(prompt, do_sample=True)
    return output[0]["generated_text"]


# -----------------------------
# Unified Interface
# -----------------------------
def generate_response(prompt: str) -> Tuple[str, str]:
    """
    Returns:
        answer (str)
        backend_used (str)
    Priority:
        1. Ollama (Mistral)
        2. HuggingFace (distilgpt2)
    """
    if ollama_available():
        return ollama_generate(prompt), "ollama"
    else:
        return hf_generate(prompt), "huggingface"