# Clasificador CIUO.08CL con BERT

Clasifica descripciones ocupacionales (glosas) según la **Clasificación Internacional Uniforme de Ocupaciones adaptada para Chile (CIUO.08CL)** usando el modelo `SABE-SENCE/CIUO08CL_4D_2024`, un BERT-base en español fine-tuneado con 161 clases.

## Especificaciones técnicas

| Atributo | Valor |
|---|---|
| Modelo | `dccuchile/bert-base-spanish-wwm-uncased` fine-tuneado |
| Arquitectura | `BertForSequenceClassification` |
| Clases | 161 códigos CIUO.08CL (4 dígitos, algunos con subclasificación `.1`, `.2`) |
| Parámetros | ~110M |
| Pesos | Safetensors F32 |
| Tokenizador | WordPiece (vocab: 31,002 tokens) |
| Contexto máx. | 512 tokens por glosa |
| Descarga automática | HuggingFace Hub (caché en `~/.cache/huggingface/`) |

### Dispositivos soportados

| Dispositivo | Detección | Rendimiento (muestras/s)* |
|---|---|---|
| GPU (CUDA) | Automática si `torch.cuda.is_available()` | ~75 glosas/s |
| CPU | Fallback automático o `--device cpu` | ~50 glosas/s |

*Medido con batch_size=32 en 15 glosas de prueba.

---

## Requisitos

- Python 3.10+
- pip

---

## Instalación

```bash
git clone <repo-url>
cd clasificador_ciuo_bert
python -m venv venv
venv/bin/pip install -r requirements.txt
```

---

## Uso

### Clasificar glosas desde CSV

```bash
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv
```

### Opciones

| Opción | Descripción |
|---|---|
| `-i, --input` | Ruta al CSV de entrada (columnas: `id`, `glosa`) |
| `-o, --output` | Ruta al CSV de salida |
| `--device {cpu,cuda}` | Forzar dispositivo (default: auto-detecta CUDA) |
| `--batch-size N` | Tamaño del lote de inferencia (default: 32) |
| `--threshold F` | Confianza mínima 0-1. Bajo este umbral se marca como 99 (default: 0.0) |
| `--show-labels` | Muestra los 161 códigos CIUO.08CL disponibles y sale |

### Ejemplos

```bash
# GPU (auto)
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv

# CPU forzado
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv --device cpu

# Lote pequeño + threshold de confianza
venv/bin/python -m src.main -i data/glosas_prueba.csv -o build/resultado.csv --batch-size 8 --threshold 0.5

# Ver todos los códigos disponibles
venv/bin/python -m src.main --show-labels
```

---

## Formato de entrada

Archivo CSV con dos columnas:

```csv
id,glosa
1,Médico cirujano en hospital público
2,Profesor de enseñanza básica en escuela municipal
```

## Formato de salida

```csv
id,glosa,gran_grupo,subgrupo_principal,subgrupo,grupo_primario,confianza
1,Médico cirujano en hospital público,2,22,222,2221,0.3281
2,Profesor de enseñanza básica en escuela municipal,2,23,233,2330,0.6131
```

| Columna | Descripción |
|---|---|
| `gran_grupo` | 1 dígito (0-9) |
| `subgrupo_principal` | 2 dígitos |
| `subgrupo` | 3 dígitos |
| `grupo_primario` | Código completo (4 dígitos, puede incluir subclasificación como `1330.1`) |
| `confianza` | Probabilidad softmax de la clase predicha (0-1) |

`"99"` en cualquier nivel jerárquico indica que no se pudo clasificar (se usa cuando la confianza está bajo el `--threshold`).

---

## Modelo

### Arquitectura

```
Input IDs (tokenized) → BERT encoder (12 layers, 768 hidden) →
[CLS] token → Linear(768, 161) → Logits → Softmax → CIUO.08CL code
```

### Códigos CIUO.08CL (161 clases)

Los 10 grandes grupos:

| Código | Nombre |
|---|---|
| 1 | Directores, Gerentes y Administradores |
| 2 | Profesionales Científicos e Intelectuales |
| 3 | Técnicos y Profesionales de Nivel Medio |
| 4 | Personal de Apoyo Administrativo |
| 5 | Trabajadores de los Servicios y Vendedores |
| 6 | Agricultores y Trabajadores Calificados |
| 7 | Oficiales, Operarios y Artesanos |
| 8 | Operadores de Instalaciones, Máquinas y Ensambladores |
| 9 | Ocupaciones Elementales |

Para listar todos los códigos con su jerarquía:

```bash
venv/bin/python -m src.main --show-labels
```

---

## Estructura del proyecto

```
clasificador_ciuo_bert/
├── data/
│   └── glosas_prueba.csv     # Base de prueba (15 glosas)
├── src/
│   ├── __init__.py
│   ├── model.py              # Carga del modelo BERT + inference
│   ├── classifier.py         # Orquestador: batch + jerarquía
│   ├── csv_handler.py        # Lectura/escritura CSV
│   └── main.py               # CLI
├── build/                    # Directorio de salida (gitignorado)
├── requirements.txt
├── instrucciones.md
└── .gitignore
```

---

## Pipeline de inferencia

1. **Lectura**: `csv_handler.read_input()` → lista de `{id, glosa}`
2. **Tokenización**: `AutoTokenizer` → input_ids, attention_mask (padding batch, truncation a 512)
3. **Forward pass**: `BertForSequenceClassification` → logits (161)
4. **Softmax**: `torch.softmax(logits)` → probabilidades por clase
5. **Top-1**: selecciona la clase con mayor probabilidad
6. **Threshold**: si prob < `threshold` → marca como `99`
7. **Jerarquía**: extrae gran_grupo/subgrupo_principal/subgrupo/grupo_primario del código
8. **Escritura**: `csv_handler.write_output()` → CSV con resultados

---

## Base de prueba

`data/glosas_prueba.csv` contiene 15 glosas que cubren:

- Profesiones universitarias (médico, abogado, ingeniero, profesor, contador)
- Técnicos (electricista, reparador de computadores)
- Administrativos (secretaria)
- Servicios (vendedor, guardia, cajero, jardinero)
- Operarios (conductor de camión)
- Caso genérico (trabajador)

---

## Notas

- El modelo se descarga automáticamente desde HuggingFace Hub en la primera ejecución y se cachea en `~/.cache/huggingface/`. No requiere conexión a internet en ejecuciones posteriores.
- El límite de 512 tokens por glosa es inherente a BERT-base. Glosas más largas se truncan automáticamente.
- El umbral de confianza (`--threshold`) permite filtrar predicciones de baja confianza marcándolas como `99`. Sin threshold (default 0.0), siempre se entrega la mejor predicción del modelo.
- Los códigos con subclasificación (e.g., `1330.1`, `2511.3`) son variantes chilenas del estándar ISCO-08 y se entregan completos en `grupo_primario`.
