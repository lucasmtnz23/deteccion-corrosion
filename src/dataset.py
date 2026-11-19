"""Carga de Corrosion-bi3q3 para los detectores de torchvision."""

from math import isfinite
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "corrosion-bi3q3"
PARTICIONES = ("train", "validation", "test")

# Los IDs originales 1, 2 y 3 ya son consecutivos. El ID original 0
# (corrosion-0) no tiene anotaciones y la etiqueta 0 se reserva para el fondo.
NOMBRES_CLASES = {1: "Slippage", 2: "corrosion", 3: "crack"}


class CorrosionDataset(Dataset):
    """Devuelve pares de imagen RGB y anotaciones para una partición original.

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
        # Los IDs de origen empiezan en cero en cada partición.
        anteriores = PARTICIONES[:PARTICIONES.index(particion)]
        self.desplazamiento_id = sum(len(datos[nombre]) for nombre in anteriores)

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
            "image_id": torch.tensor(id_original + self.desplazamiento_id, dtype=torch.int64),
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
