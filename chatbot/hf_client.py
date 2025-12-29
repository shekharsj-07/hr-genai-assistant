from transformers import pipeline

# Load
_hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

def generate(prompt: str) -> str:
    output = _hf_pipeline(prompt)
    return output[0]["generated_text"]
