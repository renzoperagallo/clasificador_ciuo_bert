import time
import logging
from src.model import CIUOBertClassifier
from src.csv_handler import read_input, write_output

logger = logging.getLogger(__name__)


def _extract_hierarchy(label: str) -> dict:
    # Handle dot notation: e.g. "1330.1" -> full code is the label itself
    # gran_grupo = first digit, subgrupo_principal = first 2, subgrupo = first 3
    clean = label.replace(".", "")
    gg = clean[0] if len(clean) >= 1 else "9"
    sp = clean[:2] if len(clean) >= 2 else "99"
    sg = clean[:3] if len(clean) >= 3 else "999"
    return {
        "gran_grupo": gg,
        "subgrupo_principal": sp,
        "subgrupo": sg,
        "grupo_primario": label,
    }


FALLBACK = {
    "gran_grupo": "99",
    "subgrupo_principal": "99",
    "subgrupo": "99",
    "grupo_primario": "99",
}


def classify_csv(
    input_path: str,
    output_path: str,
    device: str | None = None,
    batch_size: int = 32,
    threshold: float = 0.0,
):
    rows = read_input(input_path)
    total = len(rows)
    if not rows:
        logger.error("El CSV de entrada está vacío")
        return

    texts = [r["glosa"] for r in rows]
    logger.info(f"Glosas cargadas: {total}")

    classifier = CIUOBertClassifier(device=device)
    logger.info("Clasificando...")
    t0 = time.time()

    predictions = classifier.classify(texts, batch_size=batch_size, threshold=threshold)

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    logger.info(f"Clasificación completada en {elapsed:.1f}s ({rate:.1f} glosas/s)")

    output = []
    for row, pred in zip(rows, predictions):
        if pred["bajo_confianza"]:
            output.append({
                "id": row["id"],
                "glosa": row["glosa"],
                **FALLBACK,
                "confianza": f"{pred['score']:.4f}",
            })
        else:
            hierarchy = _extract_hierarchy(pred["label"])
            output.append({
                "id": row["id"],
                "glosa": row["glosa"],
                **hierarchy,
                "confianza": f"{pred['score']:.4f}",
            })

    write_output(output_path, output)
    logger.info(f"Resultados guardados en {output_path}")
