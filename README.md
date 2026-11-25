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

Abre el notebook en Jupyter o en un editor compatible y selecciona el Python de `.venv` (`.venv/bin/python` en Linux y macOS, `.venv/Scripts/python.exe` en Windows). Ejecuta las celdas en orden hasta la sección 5 para reproducir las figuras, la comparación en validación y la medición pareada de TTA. Para completar las tablas de las variantes añadidas, ejecuta también los [experimentos cruzados](#experimentos-cruzados-en-validación). Las figuras se guardan en `results/slippage_validation_split/figuras_retinanet/`.

Una vez fijada la configuración en validación, evalúa en test el checkpoint seleccionado de E2 con una vista y con TTA. El segundo comando aplica la vista original y el volteo horizontal con WBF a IoU 0,55. Con `--particion test`, el subcomando `tta` escribe `metricas_test_final.json` y `predicciones_test.json` en el directorio indicado.

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
  --particion test \
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

# Evaluación detallada de los cinco experimentos en validación
for experimento in \
  exp01_data_augmentation \
  exp02_transfer_learning \
  exp03_sgd_multistep \
  exp04_anchors_resolucion \
  exp05_perdidas_ciou_ponderada
do
  salida="results/slippage_validation_split/${experimento}"

  uv run --locked python -m src.evaluar \
    --checkpoint "${salida}/mejor_modelo.pth" \
    --particion validation \
    --batch-size 4 \
    --guardar "${salida}/metricas_validation.json" \
    --guardar-preds "${salida}/predicciones_validation.json"
done

```

En la ronda original, E1 fue el mejor checkpoint individual por mAP global en validación. Se usó para el ajuste de NMS y para el TTA de tres vistas. E6 combina E1, E3b (anchors ajustados a 960 px) y E4 (CIoU y entropía cruzada ponderada). Los siguientes comandos reproducen el postprocesado y la evaluación de las configuraciones seleccionadas para test.

```bash
raiz_resultados="results/slippage_validation_split"
checkpoint_e1="${raiz_resultados}/exp01_data_augmentation/mejor_modelo.pth"

# E5a selecciona los umbrales en validación y evalúa la pareja elegida en test.
uv run --locked python -m scripts.evaluar_postprocesado nms \
  --checkpoint "${checkpoint_e1}" --evaluar-test

# E5b usa las tres vistas fijadas, en ambos repartos.
for particion in validation test; do
  uv run --locked python -m scripts.evaluar_postprocesado tta \
    --checkpoint "${checkpoint_e1}" --particion "${particion}" \
    --vistas original hflip vflip --iou-wbf 0.55
done

# Evaluación final del baseline y de los tres componentes de E6.
for experimento in \
  faster_rcnn \
  exp01_data_augmentation \
  exp04_anchors_resolucion \
  exp05_perdidas_ciou_ponderada
do
  salida="results/slippage_validation_split/${experimento}"

  uv run --locked python -m src.evaluar \
    --checkpoint "${salida}/mejor_modelo.pth" \
    --particion test \
    --batch-size 4 \
    --guardar "${salida}/metricas_test_final.json" \
    --guardar-preds "${salida}/predicciones_test.json"
done

# Fusionar las predicciones de E6 con la configuración seleccionada.
for particion in validation test; do
  uv run --locked python -m scripts.evaluar_postprocesado ensemble \
    --particion "${particion}" \
    --preds "${raiz_resultados}/exp01_data_augmentation/predicciones_${particion}.json" \
            "${raiz_resultados}/exp04_anchors_resolucion/predicciones_${particion}.json" \
            "${raiz_resultados}/exp05_perdidas_ciou_ponderada/predicciones_${particion}.json" \
    --pesos 1.0 1.0 1.0 --iou-wbf 0.55 \
    --salida "${raiz_resultados}/exp08_ensemble_wbf"
done
```

El [notebook de Faster R-CNN](notebooks/resultados_faster_rcnn.ipynb) conserva las salidas de la ronda original y muestra aparte la ampliación. El E1 reentrenado que sirvió de base para el TTA de dos vistas obtuvo 0,4685 de mAP en validación. El E1 de la ronda original obtuvo 0,4854. Para reproducir las cifras exactas de aquella ronda, usa sus predicciones o checkpoints.

### Experimentos cruzados en validación

Después de preparar el entorno y el dataset, ejecuta desde la raíz del proyecto:

```bash
uv run --locked python -m scripts.comparar_experimentos --modelo ambos
```

Usa `--modelo retinanet` o `--modelo faster_rcnn` para ejecutar solo una arquitectura. El script comprueba la configuración solicitada contra `config_usada.yaml` y contra la configuración del checkpoint que va a usar. Entrena las variantes nuevas y solo reanuda cuando los checkpoints y el historial existentes permiten continuar con la misma configuración. Si detecta una incompatibilidad, se detiene antes de sobrescribir el entrenamiento. Usa otro directorio de salida para la nueva configuración.

- **RetinaNet** prueba descongelado progresivo, SGD con MultiStepLR, anchors ajustados a 960 px y CIoU con focal ponderada. También evalúa NMS, TTA de dos y tres vistas sobre E2 a 20 épocas, y la fusión de tres RetinaNet (aumentos, anchors ajustados y pérdida ponderada).
- **Faster R-CNN** prueba 960 px con anchors originales a 12 y 20 épocas, SGD con coseno y TTA de dos vistas sobre el E1 disponible.

El script reutiliza cada evaluación o postprocesado si existen `metricas_validation.json` y `predicciones_validation.json`. Si falta alguno, vuelve a generar ambos archivos. Al entrenar o reanudar un modelo, también regenera su evaluación y el postprocesado que depende de él.

Usa `--solo-evaluacion` para trabajar con los checkpoints disponibles. Esta opción exige entrenamientos completos y compatibles e impide entrenar o reanudar. Si cambias los checkpoints, la partición del dataset, el evaluador o los parámetros o entradas del postprocesado, usa `--forzar-evaluacion` para recalcular y reemplazar los JSON de validación de la arquitectura seleccionada. Puedes combinar ambas opciones.

```bash
uv run --locked python -m scripts.comparar_experimentos --modelo ambos --solo-evaluacion
uv run --locked python -m scripts.comparar_experimentos --modelo ambos --solo-evaluacion --forzar-evaluacion
```

Para reproducir la prueba adicional en test del checkpoint individual de Faster R-CNN seleccionado tras la ampliación, ejecuta:

```bash
salida="results/slippage_validation_split/faster_rcnn_resolucion_20ep"
uv run --locked python -m src.evaluar \
  --checkpoint "${salida}/mejor_modelo.pth" \
  --particion test \
  --batch-size 4 \
  --guardar "${salida}/metricas_test_final.json" \
  --guardar-preds "${salida}/predicciones_test.json"
```

### Ensemble entre Faster R-CNN y RetinaNet

El notebook del ensemble combina E6 de Faster R-CNN con RetinaNet E2 de una o dos vistas. Las dos variantes se evalúan en validación con WBF, pesos 1:1 e IoU 0,55. Solo la fusión con mayor mAP global en validación pasa a test. El notebook y el comando de ensemble alinean las predicciones con las anotaciones por `image_id` y rechazan IDs duplicados, ausentes o inesperados. El notebook guarda las métricas, predicciones fusionadas y figura en `results/slippage_validation_split/ensemble_interarquitectura/`. [Notebook del ensemble](notebooks/resultados_ensemble.ipynb)

Prepara estas entradas bajo `results/slippage_validation_split/`:

| Carpeta | Archivos de entrada |
|---|---|
| `faster_rcnn_pablo/ensemble_e6/` | `predicciones_validation.json` y `predicciones_test.json` del E6 original |
| `retinanet_exp02_resolucion_20ep/` | `predicciones_validation.json`, `predicciones_test.json` y `metricas_test_final.json` de E2 con una vista |
| `retinanet_exp02_resolucion_20ep_tta/` | `predicciones_validation.json` y `predicciones_test.json` de E2 con dos vistas |
| `faster_rcnn_resolucion_20ep/` | `metricas_test_final.json` para la comparación de checkpoints individuales |

Los JSON de referencia se comparten aparte, ya que Git ignora `results/` y los checkpoints. Para reproducir la fusión con un E6 regenerado, ejecuta los comandos de Faster R-CNN y copia sus predicciones a la ruta que lee el notebook:

```bash
raiz_resultados="results/slippage_validation_split"
destino="${raiz_resultados}/faster_rcnn_pablo/ensemble_e6"
mkdir -p "${destino}"
cp -i "${raiz_resultados}/exp08_ensemble_wbf/predicciones_validation.json" "${destino}/"
cp -i "${raiz_resultados}/exp08_ensemble_wbf/predicciones_test.json" "${destino}/"
```

Con los archivos colocados, abre el notebook con el intérprete de `.venv` y ejecuta las celdas en orden. La evaluación de la fusión usa las predicciones guardadas y las anotaciones del reparto corregido. En la ejecución documentada, E6 solo obtuvo el mayor mAP global tanto en validación (0,5146) como en test (0,4696). La mejor fusión fue la de dos vistas (0,5067 y 0,4657).

---
