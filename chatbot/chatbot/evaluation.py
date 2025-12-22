import mlflow
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class RAGEvaluator:
    def __init__(self, experiment_name="hr_genai_rag"):
        # 🔑 IMPORTANT
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

        self.rouge = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )
        self.smooth_fn = SmoothingFunction().method1

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_text: str,
        model_backend: str,
    ):
        with mlflow.start_run():
            rouge_l = self.rouge.score(
                reference_text, answer
            )["rougeL"].fmeasure

            bleu = sentence_bleu(
                [reference_text.split()],
                answer.split(),
                smoothing_function=self.smooth_fn,
            )

            mlflow.log_metric("rougeL_f1", rouge_l)
            mlflow.log_metric("bleu", bleu)

            mlflow.log_param("model_backend", model_backend)
            mlflow.log_param("question", question)

            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text(reference_text, "retrieved_context.txt")

            return {
                "rougeL_f1": rouge_l,
                "bleu": bleu,
            }