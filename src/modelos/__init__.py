"""Registro y utilidades de construcción de modelos de detección."""

from __future__ import annotations

import torch


def contar_parametros(modelo: torch.nn.Module) -> tuple[int, int]:
    """Devuelve ``(parametros_totales, parametros_entrenables)``."""
    totales = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return totales, entrenables


def crear_modelo(nombre: str, **kwargs):
    """Construye un detector por nombre."""
    nombre_normalizado = nombre.strip().lower()
    if nombre_normalizado == "faster_rcnn":
        from src.modelos.faster_rcnn import crear_faster_rcnn

        return crear_faster_rcnn(**kwargs)
    if nombre_normalizado == "retinanet":
        from src.modelos.retinanet import crear_retinanet

        return crear_retinanet(**kwargs)
    raise ValueError(f"Modelo desconocido: {nombre!r}. Disponibles: 'faster_rcnn', 'retinanet'.")


__all__ = [
    "contar_parametros",
    "crear_modelo",
]
