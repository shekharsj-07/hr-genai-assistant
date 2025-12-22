import subprocess
import time
import requests


OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "mistral"


def is_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server():
    """
    Start Ollama server in background.
    """
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama is not installed. Please install from https://ollama.com/download"
        )


def ensure_model_available(model_name: str = MODEL_NAME):
    """
    Ensure the required model is available locally.
    """
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )

    if model_name not in result.stdout:
        subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
        )


def ensure_ollama_ready():
    """
    Full bootstrap:
    - Start server if not running
    - Pull model if missing
    """
    if not is_ollama_running():
        start_ollama_server()

    ensure_model_available()