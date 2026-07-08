# Clasificador CIUO.08CL con BERT

Clasifica descripciones ocupacionales (glosas) en códigos **CIUO.08CL** usando el modelo `SABE-SENCE/CIUO08CL_4D_2024`, un BERT-base en español (`dccuchile/bert-base-spanish-wwm-uncased`) fine-tuneado con **161 clases**. Ejecuta inferencia local en **CPU o GPU** con detección automática de dispositivo.

## Requisitos

- Python 3.10+
- pip

## Instalación

```bash
git clone <repo-url>
cd clasificador_ciuo_bert
python -m venv venv
venv/bin/pip install -r requirements.txt
```

El modelo (~440 MB) se descarga automáticamente desde HuggingFace Hub en la primera ejecución y queda cacheado en `~/.cache/huggingface/`.

## Uso

```bash
# Clasificar glosas (auto-detecta CUDA si está disponible)
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv

# Forzar CPU
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv --device cpu

# Umbral de confianza (bajo 0.5 se marca como 99)
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv --threshold 0.5

# Listar los 161 códigos disponibles
venv/bin/python -m src.main --show-labels
```

### Opciones

| Opción | Descripción |
|---|---|
| `-i, --input` | CSV de entrada (columnas: `id`, `glosa`) |
| `-o, --output` | CSV de salida |
| `--device {cpu,cuda}` | Forzar dispositivo (default: auto) |
| `--batch-size N` | Lote de inferencia (default: 32) |
| `--threshold F` | Confianza mínima 0-1 (default: 0.0 = siempre usar la mejor) |
| `--show-labels` | Lista los 161 códigos y sale |

## Formato

**Entrada:**
```csv
id,glosa
1,Médico cirujano en hospital público
```

**Salida:**
```csv
id,glosa,gran_grupo,subgrupo_principal,subgrupo,grupo_primario,confianza
1,Médico cirujano en hospital público,2,22,222,2221,0.3281
```

- `gran_grupo` (1 digito), `subgrupo_principal` (2), `subgrupo` (3), `grupo_primario` (codigo completo)
- `confianza`: probabilidad softmax de la clase predicha
- `"99"` en cualquier nivel = no se pudo clasificar (se activa con `--threshold`)

## Arquitectura

```
Tokenizer → BERT encoder (12 capas, 768 hidden) → [CLS] → Linear(768, 161) → Softmax → codigo CIUO.08CL
```

## Resultados de prueba

| Dispositivo | Velocidad |
|---|---|
| GPU (CUDA) | ~75 glosas/s |
| CPU | ~50 glosas/s |

Medido con batch_size=32 en 15 glosas.

## Estructura

```
clasificador_ciuo_bert/
├── src/
│   ├── model.py        # Carga del modelo + inference
│   ├── classifier.py   # Orquestador por lotes + jerarquia
│   ├── csv_handler.py  # I/O de CSV
│   └── main.py         # CLI
├── data/
│   └── glosas_prueba.csv  # 15 glosas de prueba
├── requirements.txt
├── instrucciones.md    # Especificacion tecnica detallada
└── README.md
```

Para especificaciones técnicas detalladas ver `instrucciones.md`.
