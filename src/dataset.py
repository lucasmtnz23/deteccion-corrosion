"""Carga de Corrosion-bi3q3 para los detectores de torchvision."""

import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import Dataset


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ_PROYECTO / "data" / "corrosion-bi3q3"
RUTA_MANIFIESTO = RAIZ_PROYECTO / "configs" / "particiones" / "slippage_validation_split.json"
PARTICIONES = ("train", "validation", "test")

# Los IDs originales 1, 2 y 3 ya son consecutivos. El ID original 0
# (corrosion-0) no tiene anotaciones y la etiqueta 0 se reserva para el fondo.
NOMBRES_CLASES = {1: "Slippage", 2: "corrosion", 3: "crack"}


def _indices_trasladados(train):
    with RUTA_MANIFIESTO.open(encoding="utf-8") as archivo:
        ids_manifiesto = json.load(archivo)["image_ids_train"]
    ids_trasladados = set(ids_manifiesto)
    indices = [i for i, id_original in enumerate(train["image_id"]) if id_original in ids_trasladados]
    if not len(ids_manifiesto) == len(ids_trasladados) == len(indices) == 20:
        raise ValueError("El manifiesto debe identificar 20 imágenes únicas del train original.")
    if any(1 not in objetos["category"] for objetos in train.select(indices)["objects"]):
        raise ValueError("Todas las imágenes del manifiesto deben contener Slippage.")
    return indices


class CorrosionDataset(Dataset):
    """Devuelve pares de imagen RGB y anotaciones para una partición corregida.

    La transformación opcional recibe ``(imagen, anotaciones)`` tras convertir
    ambos a tensores y devuelve el par transformado. Si cambia la geometría,
    debe actualizar también las cajas y sus áreas.
    """

    def __init__(self, particion="train", transformacion=None):
        if particion not in PARTICIONES:
            raise ValueError(f"Partición {particion!r} inválida. Usa {', '.join(PARTICIONES)}.")

        datos = load_from_disk(str(RUTA_DATOS))
        self.datos = datos[particion]
        self.particion = particion
        self.transformacion = transformacion
        # Los desplazamientos se calculan con las particiones de origen para conservar los IDs.
        anteriores = PARTICIONES[:PARTICIONES.index(particion)]
        desplazamiento = sum(len(datos[nombre]) for nombre in anteriores)
        self._ids_evaluacion = [id_original + desplazamiento for id_original in self.datos["image_id"]]

        if particion != "test":
            indices_trasladados = _indices_trasladados(datos["train"])
            if particion == "train":
                trasladados = set(indices_trasladados)
                restantes = [i for i in range(len(self.datos)) if i not in trasladados]
                self.datos = self.datos.select(restantes)
                self._ids_evaluacion = [self._ids_evaluacion[i] for i in restantes]
            else:
                trasladadas = datos["train"].select(indices_trasladados)
                self.datos = concatenate_datasets([self.datos, trasladadas])
                # Las filas trasladadas conservan el desplazamiento cero de train.
                self._ids_evaluacion.extend(trasladadas["image_id"])

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, indice):
        ejemplo = self.datos[indice]
        imagen_pil = ejemplo["image"].convert("RGB")
        ancho_imagen, alto_imagen = imagen_pil.size
        id_original = ejemplo["image_id"]
        contexto = f"Partición {self.particion}, imagen original {id_original}"
        objetos = ejemplo["objects"]
        cajas_originales = objetos["bbox"]
        etiquetas = objetos["category"]
        if len(cajas_originales) != len(etiquetas):
            raise ValueError(f"{contexto}: las cantidades de cajas y etiquetas difieren.")

        cajas = []
        for numero, (caja, etiqueta) in enumerate(zip(cajas_originales, etiquetas)):
            if etiqueta not in NOMBRES_CLASES:
                raise ValueError(f"{contexto}, objeto {numero}: categoría inesperada {etiqueta!r}.")

            x, y, ancho, alto = caja
            x_fin, y_fin = x + ancho, y + alto
            if not (
                all(map(isfinite, (x, y, x_fin, y_fin)))
                and 0 <= x < x_fin <= ancho_imagen
                and 0 <= y < y_fin <= alto_imagen
            ):
                raise ValueError(
                    f"{contexto}, objeto {numero}: caja COCO inválida {caja!r} "
                    f"para una imagen de {ancho_imagen}×{alto_imagen} píxeles."
                )
            cajas.append((x, y, x_fin, y_fin))

        cajas = torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4)
        dimensiones = cajas[:, 2:] - cajas[:, :2]
        degeneradas = (dimensiones <= 0).any(dim=1).nonzero().flatten()
        if degeneradas.numel():
            numero = degeneradas[0].item()
            raise ValueError(f"{contexto}, objeto {numero}: la caja pierde tamaño en float32.")

        anotaciones = {
            "boxes": cajas,
            "labels": torch.tensor(etiquetas, dtype=torch.int64),
            "image_id": torch.tensor(self._ids_evaluacion[indice], dtype=torch.int64),
            "area": dimensiones[:, 0] * dimensiones[:, 1],
            "iscrowd": torch.zeros(len(cajas), dtype=torch.int64),
        }
        imagen = torch.from_numpy(np.array(imagen_pil)).permute(2, 0, 1)
        imagen = imagen.to(dtype=torch.float32).div_(255)

        if self.transformacion is not None:
            return self.transformacion(imagen, anotaciones)
        return imagen, anotaciones


def collate_fn(lote):
    """Agrupa por separado imágenes y anotaciones, conservando sus tamaños."""
    imagenes, anotaciones = zip(*lote)
    return imagenes, anotaciones
