import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_NAME = "SABE-SENCE/CIUO08CL_4D_2024"

logger = logging.getLogger(__name__)


def detect_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        logger.info("CUDA detectado — usando GPU")
        return torch.device("cuda")
    logger.info("CUDA no disponible — usando CPU")
    return torch.device("cpu")


class CIUOBertClassifier:
    def __init__(self, device: str | None = None):
        self.device = detect_device(device)
        logger.info(f"Cargando tokenizador desde {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Cargando modelo desde {MODEL_NAME}...")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            .to(self.device)
            .eval()
        )
        self.id2label = self.model.config.id2label
        logger.info(
            f"Modelo cargado en {self.device} | "
            f"{len(self.id2label)} clases"
        )

    @torch.no_grad()
    def classify(
        self,
        texts: list[str],
        batch_size: int = 32,
        threshold: float = 0.0,
    ) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            for j in range(len(batch_texts)):
                score = top_probs[j].item()
                label_idx = int(top_indices[j].item())
                label_id = self.model.config.id2label.get(label_idx, self.model.config.id2label.get(str(label_idx)))
                if label_id is None:
                    label = "99"
                else:
                    label = label_id
                results.append({
                    "label": label,
                    "score": score,
                    "bajo_confianza": score < threshold,
                })
        return results
