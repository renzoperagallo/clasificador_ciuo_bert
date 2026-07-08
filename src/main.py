import argparse
import logging
import sys
from src.classifier import classify_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)


def main():
    parser = argparse.ArgumentParser(
        description="Clasificador CIUO.08CL con BERT — "
                    "Clasifica descripciones ocupacionales usando "
                    "SABE-SENCE/CIUO08CL_4D_2024"
    )
    parser.add_argument(
        "-i", "--input",
        help="Ruta al CSV de entrada (columnas: id, glosa)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Ruta al CSV de salida",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia (default: auto-detecta CUDA)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del lote de inferencia (default: 32)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Confianza mínima (0-1). Bajo este umbral se marca como 99 "
             "(default: 0.0 = siempre usar la mejor predicción)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Muestra todos los códigos CIUO.08CL disponibles y sale",
    )

    args = parser.parse_args()

    if args.show_labels:
        from src.model import CIUOBertClassifier
        c = CIUOBertClassifier(device=args.device)
        codes = sorted(c.id2label.values(), key=lambda x: (
            int(x.split(".")[0]) if x.split(".")[0].isdigit() else 9999,
            x,
        ))
        print(f"{'Código':<12} Jerarquía")
        print("-" * 50)
        for code in codes:
            from src.classifier import _extract_hierarchy
            h = _extract_hierarchy(code)
            print(f"{code:<12} {h['gran_grupo']} › {h['subgrupo_principal']} › {h['subgrupo']} › {h['grupo_primario']}")
        print(f"\nTotal: {len(codes)} códigos")
        return

    classify_csv(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
