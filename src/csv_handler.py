import csv
from pathlib import Path


FIELD_NAMES = [
    "id",
    "glosa",
    "gran_grupo",
    "subgrupo_principal",
    "subgrupo",
    "grupo_primario",
    "confianza",
]


def read_input(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "id": row["id"].strip(),
                "glosa": row["glosa"].strip(),
            })
    return rows


def write_output(csv_path: str, rows: list[dict]):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)
