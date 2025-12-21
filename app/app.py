import sys
from pathlib import Path
import os

# -----------------------------
# PYTHONPATH FIX (MUST BE FIRST)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional: silence locale warning
os.environ["CHAINLIT_LANG"] = "en-US"

# -----------------------------
# OPTIONAL: Ensure Ollama ready
# -----------------------------
from chatbot.ollama_utils import ensure_ollama_ready
ensure_ollama_ready()

# -----------------------------
# Normal imports AFTER path fix
# -----------------------------
import chainlit as cl

from chatbot.faq_insights import FAQInsights
from chatbot.loader import MultiDocumentLoader
from chatbot.chunking import DocumentChunker
from chatbot.vectorstore import VectorStoreManager
from chatbot.rag_chain import HRPolicyRAG
from chatbot.history import ChatHistoryStore
from chatbot.evaluation import RAGEvaluator


# GLOBAL INITIALIZATION

loader = MultiDocumentLoader()
docs = loader.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(docs)

vs_manager = VectorStoreManager()
vectorstore = vs_manager.get_or_create(chunks)

rag = HRPolicyRAG(vectorstore)
history_store = ChatHistoryStore()
evaluator = RAGEvaluator()

#----------------------------
# CHAINLIT EVENTS
#---------------------------

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "👋 **Welcome to the Acme HR Policy Assistant**\n\n"
            "Ask questions about:\n"
            "- Leave policies\n"
            "- Working hours\n"
            "- Benefits\n"
            "- Code of conduct\n\n"
            "_Each session is fresh. Responses are evaluated and logged._"
            )
    ).send()

        # --- FAQ Insights ---
    past_questions = history_store.fetch_all_questions()
    faq = FAQInsights(past_questions).top_faqs()

    if faq:
        faq_text = "📌 **Most Frequently Asked Questions**\n\n"
        for q, count in faq.items():
            faq_text += f"• {q} _(asked {count} times)_\n"

        await cl.Message(
            content=faq_text,
            author="Insights"
        ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    question = message.content

    await cl.Message(content="Thinking...").send()

    # --- RAG Answer ---
    response = rag.answer(question)
    answer = response["answer"]

    # --- Reference text for evaluation ---
    reference_text = "\n".join(
        doc.page_content for doc in response["sources"]
    )

    # --- MLflow Evaluation ---
    metrics = evaluator.evaluate(
        question=question,
        answer=answer,
        reference_text=reference_text,
        model_backend="ollama_or_hf_fallback",
    )

    # --- Log history ---
    history_store.log(question, answer)

    # --- Final response to UI ---
    await cl.Message(
        content=(
            f"{answer}\n\n"
            f"📊 **Evaluation Metrics**\n"
            f"- ROUGE-L F1: `{metrics['rougeL_f1']:.3f}`\n"
            f"- BLEU: `{metrics['bleu']:.3f}`"
        )
    ).send()


def fetch_all_questions(self):
    cursor = self.conn.cursor()
    cursor.execute("SELECT question FROM chat_history")
    rows = cursor.fetchall()
    return [r[0] for r in rows]