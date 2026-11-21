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

Contiene 1.249 imágenes divididas en:

| Partición | Imágenes |
|---|---:|
| Entrenamiento (`train`) | 840 |
| Validación (`validation`) | 105 |
| Test (`test`) | 304 |

Para descargarlo, ejecutar:

```bash
uv run python scripts/descargar_datos.py
```

El script descarga el dataset desde Hugging Face y lo guarda en `data/corrosion-bi3q3/`.

La carpeta `data/` está incluida en `.gitignore`.

### 4. Cargar un lote de imágenes y anotaciones

Ejecuta este ejemplo desde la raíz del proyecto con `uv run python`. Puedes seleccionar cualquiera de las tres particiones de la tabla.

```python
from torch.utils.data import DataLoader

from src.dataset import CorrosionDataset, collate_fn

datos = CorrosionDataset("train")
cargador = DataLoader(datos, batch_size=2, collate_fn=collate_fn)
imagenes, anotaciones = next(iter(cargador))
```

Cada imagen es un tensor RGB `float32` con forma `[3, H, W]` y valores entre 0 y 1. Cada anotación contiene las cajas en formato `xyxy` y sus etiquetas. `collate_fn` conserva cada imagen y su diccionario de anotaciones, de modo que un lote admite cantidades distintas de objetos por imagen, incluidas imágenes sin objetos.

El mapeo de etiquetas es `0` para fondo, `1` para `Slippage`, `2` para `corrosion` y `3` para `crack`. La categoría original `corrosion-0` carece de anotaciones en las tres particiones.

---
