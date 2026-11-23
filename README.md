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
uv sync --locked
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
uv run --locked python scripts/descargar_datos.py
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

Los checkpoints, historiales y métricas del reparto original deben regenerarse con el reparto corregido. Las configuraciones y notebooks usan `results/slippage_validation_split/` para estas ejecuciones.

Para Faster R-CNN v2, entrena el baseline y sigue las instrucciones de su [notebook](notebooks/resultados_faster_rcnn.ipynb).

```bash
uv run --locked python -m src.entrenar --config configs/faster_rcnn.yaml --sin-test
```

#### RetinaNet v2

Desde la raíz del repositorio, con el entorno y el dataset preparados, ejecuta los bloques en Bash para entrenar el baseline y las cuatro variantes. Usa `--sin-test` para reservar la partición de test hasta la evaluación final.

Cada YAML fija la semilla 42. El entrenamiento aplica esa semilla a Python, NumPy y PyTorch, incluida CUDA cuando está disponible. El hardware y las operaciones de GPU pueden introducir pequeñas diferencias entre ejecuciones. Los tiempos dependen también del dispositivo.

```bash
uv run --locked python -m src.entrenar --config configs/retinanet.yaml --sin-test

for experimento in retinanet_exp01_aumentos retinanet_exp02_resolucion retinanet_exp02_resolucion_20ep retinanet_exp03_sgd; do
  uv run --locked python -m src.entrenar --config "configs/experimentos/${experimento}.yaml" --sin-test
done
```

Genera las métricas detalladas y las predicciones de los mejores checkpoints en validación. El [notebook de RetinaNet](notebooks/resultados_retinanet.ipynb) lee estos resultados. Su sección 5 mide el efecto del TTA sobre el checkpoint de E2 a 960 px y 20 épocas, con el mismo protocolo de latencia para las dos variantes.

```bash
for experimento in retinanet retinanet_exp01_aumentos retinanet_exp02_resolucion retinanet_exp02_resolucion_20ep retinanet_exp03_sgd; do
  salida="results/slippage_validation_split/${experimento}"
  uv run --locked python -m src.evaluar \
    --checkpoint "${salida}/mejor_modelo.pth" \
    --particion validation \
    --batch-size 4 \
    --guardar "${salida}/metricas_validation.json" \
    --guardar-preds "${salida}/predicciones_validation.json"
done
```

Abre el notebook en Jupyter o en un editor compatible y selecciona el Python de `.venv` (`.venv/bin/python` en Linux y macOS, `.venv/Scripts/python.exe` en Windows). Ejecuta las celdas en orden hasta la sección 5 para reproducir las figuras, la comparación en validación y la prueba de TTA. Las figuras se guardan en `results/slippage_validation_split/figuras_retinanet/`.

Una vez fijada la configuración en validación, evalúa en test el checkpoint seleccionado de E2 con una vista y con TTA. El segundo comando aplica la vista original y el volteo horizontal con WBF a IoU 0,55. El CLI de TTA usa test y escribe `metricas_test.json` y `predicciones_test.json` en el directorio indicado.

```bash
salida="results/slippage_validation_split/retinanet_exp02_resolucion_20ep"
uv run --locked python -m src.evaluar \
  --checkpoint "${salida}/mejor_modelo.pth" \
  --particion test \
  --batch-size 4 \
  --guardar "${salida}/metricas_test_final.json" \
  --guardar-preds "${salida}/predicciones_test.json"

uv run --locked python -m scripts.evaluar_postprocesado tta \
  --checkpoint "${salida}/mejor_modelo.pth" \
  --vistas original hflip \
  --iou-wbf 0.55 \
  --salida "${salida}_tta"
```

Vuelve a ejecutar la celda de evaluación final del notebook para cargar los dos JSON de test. La vista única mide el pase del modelo en lotes de cuatro y omite el primer lote. TTA mide imagen a imagen, incluye las transferencias y WBF, y omite la primera imagen. Usa la comparación de la sección 5 del notebook para estimar el coste adicional de TTA bajo un protocolo común.

### Experimentos de mejora y evaluación de Faster R-CNN v2

```bash
# Baseline (12 épocas) + evaluación en validación
uv run --locked python -m src.entrenar --config configs/faster_rcnn.yaml --sin-test
uv run --locked python -m src.evaluar \
  --checkpoint results/slippage_validation_split/faster_rcnn/mejor_modelo.pth \
  --particion validation \
  --guardar results/slippage_validation_split/faster_rcnn/metricas_validation.json \
  --guardar-preds results/slippage_validation_split/faster_rcnn/predicciones_validation.json

# Estrategias de mejora E1–E4 (12 épocas cada una, comparativa homogénea)
uv run --locked python -m src.entrenar --config configs/experimentos/01_data_augmentation.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/02_transfer_learning.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/03_sgd_optimizacion.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/04_anchors_resolucion.yaml --sin-test
uv run --locked python -m src.entrenar --config configs/experimentos/05_perdidas_ciou_ponderada.yaml --sin-test

# Post-procesado sobre el ganador de validación (E1): E5a (NMS/confianza) y E5b (TTA 3 vistas WBF)
uv run --locked python -m scripts.evaluar_postprocesado nms --checkpoint results/slippage_validation_split/exp01_data_augmentation/mejor_modelo.pth
uv run --locked python -m scripts.evaluar_postprocesado tta --checkpoint results/slippage_validation_split/exp01_data_augmentation/mejor_modelo.pth --particion validation --vistas original hflip vflip

# Ensemble WBF intra-arquitectura E6 (E1 + E3b + E4, pesos iguales 1:1:1)
uv run --locked python -m scripts.evaluar_postprocesado ensemble \
  --particion test \
  --preds results/slippage_validation_split/exp01_data_augmentation/predicciones_test.json \
          results/slippage_validation_split/exp04_anchors_resolucion/predicciones_test.json \
          results/slippage_validation_split/exp05_perdidas_ciou_ponderada/predicciones_test.json \
  --pesos 1.0 1.0 1.0 --iou-wbf 0.55
```

El resumen completo de resultados, figuras y análisis por estrategia se encuentra ejecutado en [`notebooks/resultados_faster_rcnn.ipynb`](notebooks/resultados_faster_rcnn.ipynb) y resumido en la presentación [`Presentacion_Faster_RCNN_Corrosion.pptx`](Presentacion_Faster_RCNN_Corrosion.pptx).

---
