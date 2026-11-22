# Práctica 5 — Detección de objetos

**Asignatura:** Visión por Computador  
**Titulación:** Grado en Inteligencia Artificial  
**Universidad:** Universidade de Santiago de Compostela (USC)  
**Curso académico:** 2026/2027

**Autores:**
- Lucas Andrea Martínez Calveiro
- Juan Pablo Pardo Pérez

## Descripción

Esta práctica consiste en adaptar, entrenar y evaluar dos modelos de detección de objetos preentrenados en MS COCO sobre un dataset específico.

Se utilizará el dataset **Corrosion-bi3q3**, que contiene imágenes de defectos en infraestructuras, y se compararán dos arquitecturas:

- **Faster R-CNN v2:** detector de dos etapas (*two-stage*).
- **RetinaNet v2:** detector de una etapa (*one-stage*).

El objetivo es establecer un rendimiento de referencia (*baseline*), experimentar con distintas estrategias de mejora y comparar los resultados de ambos modelos.

## Instalación y ejecución

El proyecto utiliza Python 3.12 y [uv](https://docs.astral.sh/uv/) para gestionar las dependencias y el entorno virtual.

### 1. Instalar uv

En Linux, macOS o WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

En Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Para comprobar que se ha instalado correctamente:

```bash
uv --version
```

### 2. Preparar el entorno

Desde la raíz del repositorio, instalar las dependencias:

```bash
uv sync
```

Este comando crea automáticamente el entorno virtual `.venv` e instala las dependencias definidas en el proyecto.

Para ejecutar scripts se utiliza `uv run`, que emplea automáticamente el entorno virtual, sin necesidad de activarlo manualmente:

```bash
uv run python ruta/al/script.py
```

### 3. Descargar el dataset

El dataset utilizado es [Corrosion-bi3q3](https://huggingface.co/datasets/Francesco/corrosion-bi3q3), perteneciente a Roboflow-100.

Para descargarlo, ejecutar:

```bash
uv run python scripts/descargar_datos.py
```

El script descarga el dataset desde Hugging Face y lo guarda en `data/corrosion-bi3q3/`.

La carpeta `data/` está incluida en `.gitignore`.

### 4. Cargar un lote de imágenes y anotaciones

Ejecuta este ejemplo desde la raíz del proyecto con `uv run python`. Puedes seleccionar `train`, `validation` o `test`.

```python
from torch.utils.data import DataLoader

from src.dataset import CorrosionDataset, collate_fn

datos = CorrosionDataset("train")
cargador = DataLoader(datos, batch_size=2, collate_fn=collate_fn)
imagenes, anotaciones = next(iter(cargador))
```

Cada imagen es un tensor RGB `float32` con forma `[3, H, W]` y valores entre 0 y 1. Cada anotación contiene las cajas en formato `xyxy` y sus etiquetas. `collate_fn` conserva cada imagen y su diccionario de anotaciones, de modo que un lote admite cantidades distintas de objetos por imagen, incluidas imágenes sin objetos.

El mapeo de etiquetas es `0` para fondo, `1` para `Slippage`, `2` para `corrosion` y `3` para `crack`. La categoría original `corrosion-0` carece de anotaciones en las tres particiones.

## Particiones del dataset

Por indicación del profesor, trasladamos 20 imágenes completas con `Slippage` del `train` original a `validation`, que originalmente tenía cero anotaciones de esa clase. Se conservan las 105 imágenes originales de validación y se añaden las 20 con todas sus anotaciones. `test` conserva sus 304 imágenes, su orden y sus anotaciones.

Para reproducir la selección, se ordenan los IDs originales de las 215 imágenes de `train` con al menos una anotación de etiqueta `1` y se toman 20 sin reemplazo con `random.Random(42).sample(ids_ordenados, 20)`. Los IDs seleccionados se guardan en el manifiesto del repositorio. [Manifiesto](configs/particiones/slippage_validation_split.json)

`CorrosionDataset` aplica el reparto en memoria al cargar los archivos originales de `data/corrosion-bi3q3/`. Los IDs de evaluación se calculan según la partición de origen, así que las imágenes trasladadas conservan su ID y el conjunto sigue siendo único entre las tres particiones.

| Partición | Imágenes original → corregido | Slippage | corrosion | crack | Total anotaciones |
|---|---:|---:|---:|---:|---:|
| `train` | 840 → 820 | 548 → 499 | 1176 → 1175 | 1388 → 1388 | 3112 → 3062 |
| `validation` | 105 → 125 | 0 → 49 | 164 → 165 | 258 → 258 | 422 → 472 |
| `test` | 304 → 304 | 98 → 98 | 317 → 317 | 867 → 867 | 1282 → 1282 |
| **Total** | **1249 → 1249** | **646 → 646** | **1657 → 1657** | **2513 → 2513** | **4816 → 4816** |

Las 20 imágenes trasladadas aportan 49 anotaciones de `Slippage` (entre 1 y 4 por imagen) y una de `corrosion`. Todos los experimentos de Faster R-CNN v2 y RetinaNet v2 usan este mismo reparto.

### Repetir los experimentos con el reparto corregido

Los checkpoints, historiales y métricas de las ejecuciones anteriores corresponden al reparto original. Hay que reentrenar ambos modelos con el reparto corregido y volver a generar las métricas y predicciones antes de comparar los resultados. Las configuraciones y notebooks usan `results/slippage_validation_split/` para las nuevas ejecuciones.

```bash
uv run --locked python -m src.entrenar --config configs/faster_rcnn.yaml
uv run --locked python -m src.entrenar --config configs/retinanet.yaml
```

Los notebooks indican cómo generar métricas detalladas de validación y repetir los experimentos de mejora y posprocesado con los nuevos checkpoints.

### Experimentos de mejora y evaluación de Faster R-CNN v2

```bash
# Baseline (12 épocas) + evaluación en validación
uv run --locked python -m src.entrenar --config configs/faster_rcnn.yaml --sin-test
uv run --locked python -m src.evaluar \
  --checkpoint results/slippage_validation_split/faster_rcnn/mejor_modelo.pth \
  --particion validation \
  --guardar results/slippage_validation_split/faster_rcnn/metricas_validation.json \
  --guardar-preds results/slippage_validation_split/faster_rcnn/predicciones_validation.json

# Estrategias de mejora E1–E4 (8 épocas cada una)
uv run --locked python -m src.entrenar --config configs/experimentos/01_data_augmentation.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/02_transfer_learning.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/03_sgd_optimizacion.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/04_anchors_resolucion.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/05_perdidas_ciou_ponderada.yaml --sin-test

# Post-procesado (E5a NMS/confianza, E5b TTA WBF) y Ensemble WBF (E6)
uv run --locked python -m scripts.evaluar_postprocesado nms --checkpoint results/slippage_validation_split/faster_rcnn/mejor_modelo.pth
uv run --locked python -m scripts.evaluar_postprocesado tta --checkpoint results/slippage_validation_split/faster_rcnn/mejor_modelo.pth --particion validation --vistas original hflip
```

El resumen completo de resultados, figuras y análisis por estrategia se encuentra ejecutado en `notebooks/resultados_faster_rcnn.ipynb`.

---
