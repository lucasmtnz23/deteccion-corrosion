"""Transformaciones y aumento de datos (Data Augmentation) para detección de objetos.

Todas las transformaciones operan sobre el par ``(imagen, anotaciones)`` producido
por ``CorrosionDataset`` (tensor RGB float32 en ``[0, 1]`` de forma ``(3, H, W)``
y diccionario con ``boxes`` en formato ``xyxy``, ``labels``, ``area``, ``iscrowd``).
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torchvision.transforms.v2.functional as F_v2


class ComponerTransformaciones:
    """Aplica secuencialmente una lista de transformaciones ``(imagen, anotaciones)``."""

    def __init__(self, transformaciones: list[Callable]):
        self.transformaciones = transformaciones

    def __call__(self, imagen: torch.Tensor, anotaciones: dict[str, torch.Tensor]):
        for transformacion in self.transformaciones:
            imagen, anotaciones = transformacion(imagen, anotaciones)
        return imagen, anotaciones


class VolteoHorizontalAleatorio:
    """Voltea horizontalmente imagen y cajas con probabilidad ``prob``."""

    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, imagen: torch.Tensor, anotaciones: dict[str, torch.Tensor]):
        if self.prob <= 0.0 or random.random() >= self.prob:
            return imagen, anotaciones

        imagen = torch.flip(imagen, dims=[2])
        anotaciones = dict(anotaciones)
        cajas = anotaciones["boxes"]
        if cajas.numel() > 0:
            ancho = float(imagen.shape[2])
            cajas = cajas.clone()
            cajas[:, [0, 2]] = ancho - cajas[:, [2, 0]]
            anotaciones["boxes"] = cajas
        return imagen, anotaciones


class VolteoVerticalAleatorio:
    """Voltea verticalmente imagen y cajas con probabilidad ``prob``.

    En inspección de superficies (grietas, corrosión, deslizamientos), los defectos
    son en gran medida invariantes a la orientación vertical.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, imagen: torch.Tensor, anotaciones: dict[str, torch.Tensor]):
        if self.prob <= 0.0 or random.random() >= self.prob:
            return imagen, anotaciones

        imagen = torch.flip(imagen, dims=[1])
        anotaciones = dict(anotaciones)
        cajas = anotaciones["boxes"]
        if cajas.numel() > 0:
            alto = float(imagen.shape[1])
            cajas = cajas.clone()
            cajas[:, [1, 3]] = alto - cajas[:, [3, 1]]
            anotaciones["boxes"] = cajas
        return imagen, anotaciones


class AjusteColorAleatorio:
    """Aplica perturbaciones fotométricas (brillo, contraste, saturación y tono)."""

    def __init__(
        self,
        brillo: float = 0.2,
        contraste: float = 0.2,
        saturacion: float = 0.2,
        tono: float = 0.04,
        prob: float = 0.7,
    ):
        self.brillo = float(brillo)
        self.contraste = float(contraste)
        self.saturacion = float(saturacion)
        self.tono = float(tono)
        self.prob = float(prob)

    def __call__(self, imagen: torch.Tensor, anotaciones: dict[str, torch.Tensor]):
        if self.prob <= 0.0 or random.random() >= self.prob:
            return imagen, anotaciones

        img = imagen
        orden = ["brillo", "contraste", "saturacion", "tono"]
        random.shuffle(orden)

        for paso in orden:
            if paso == "brillo" and self.brillo > 0:
                factor = random.uniform(max(0.0, 1.0 - self.brillo), 1.0 + self.brillo)
                img = F_v2.adjust_brightness(img, factor)
            elif paso == "contraste" and self.contraste > 0:
                factor = random.uniform(max(0.0, 1.0 - self.contraste), 1.0 + self.contraste)
                img = F_v2.adjust_contrast(img, factor)
            elif paso == "saturacion" and self.saturacion > 0:
                factor = random.uniform(max(0.0, 1.0 - self.saturacion), 1.0 + self.saturacion)
                img = F_v2.adjust_saturation(img, factor)
            elif paso == "tono" and self.tono > 0:
                factor = random.uniform(-self.tono, self.tono)
                img = F_v2.adjust_hue(img, factor)

        return img.clamp(0.0, 1.0), anotaciones


class RecorteAleatorioConCajas:
    """Recorta una subregión aleatoria y la reescala al tamaño original (zoom-in).

    Esta transformación resulta especialmente útil en Corrosion-bi3q3, donde el
    85,3 % de los objetos son pequeños (< 32² px²), ya que amplía los defectos
    pequeños durante el entrenamiento. Conserva solo cajas con al menos
    ``visibilidad_min`` de su área original dentro del recorte.
    """

    def __init__(
        self,
        escala_min: float = 0.65,
        escala_max: float = 0.95,
        visibilidad_min: float = 0.35,
        lado_min_px: float = 2.0,
        prob: float = 0.5,
        intentos: int = 8,
    ):
        self.escala_min = float(escala_min)
        self.escala_max = float(escala_max)
        self.visibilidad_min = float(visibilidad_min)
        self.lado_min_px = float(lado_min_px)
        self.prob = float(prob)
        self.intentos = int(intentos)

    def __call__(self, imagen: torch.Tensor, anotaciones: dict[str, torch.Tensor]):
        if self.prob <= 0.0 or random.random() >= self.prob:
            return imagen, anotaciones

        _, alto, ancho = imagen.shape
        cajas = anotaciones["boxes"]
        tiene_cajas = cajas.numel() > 0

        for _ in range(self.intentos):
            escala = random.uniform(self.escala_min, self.escala_max)
            nuevo_h = max(32, int(round(alto * escala)))
            nuevo_w = max(32, int(round(ancho * escala)))
            top = random.randint(0, alto - nuevo_h)
            left = random.randint(0, ancho - nuevo_w)

            if not tiene_cajas:
                recorte = imagen[:, top : top + nuevo_h, left : left + nuevo_w]
                redim = torch.nn.functional.interpolate(
                    recorte.unsqueeze(0),
                    size=(alto, ancho),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                return redim, anotaciones

            areas_orig = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
            cajas_rec = cajas.clone()
            cajas_rec[:, 0] = (cajas_rec[:, 0] - left).clamp(0.0, float(nuevo_w))
            cajas_rec[:, 2] = (cajas_rec[:, 2] - left).clamp(0.0, float(nuevo_w))
            cajas_rec[:, 1] = (cajas_rec[:, 1] - top).clamp(0.0, float(nuevo_h))
            cajas_rec[:, 3] = (cajas_rec[:, 3] - top).clamp(0.0, float(nuevo_h))

            w_rec = cajas_rec[:, 2] - cajas_rec[:, 0]
            h_rec = cajas_rec[:, 3] - cajas_rec[:, 1]
            areas_rec = w_rec * h_rec

            factor_x = float(ancho) / float(nuevo_w)
            factor_y = float(alto) / float(nuevo_h)

            validas = (
                (w_rec * factor_x >= self.lado_min_px)
                & (h_rec * factor_y >= self.lado_min_px)
                & (areas_rec >= self.visibilidad_min * areas_orig)
            )
            if not validas.any():
                continue

            cajas_final = cajas_rec[validas]
            cajas_final[:, [0, 2]] *= factor_x
            cajas_final[:, [1, 3]] *= factor_y
            cajas_final[:, [0, 2]] = cajas_final[:, [0, 2]].clamp(0.0, float(ancho))
            cajas_final[:, [1, 3]] = cajas_final[:, [1, 3]].clamp(0.0, float(alto))

            dims = cajas_final[:, 2:] - cajas_final[:, :2]
            positivas = (dims[:, 0] > 0) & (dims[:, 1] > 0)
            if not positivas.any():
                continue

            indices = torch.where(validas)[0][positivas]
            cajas_final = cajas_final[positivas]
            dims = dims[positivas]

            recorte = imagen[:, top : top + nuevo_h, left : left + nuevo_w]
            redim = torch.nn.functional.interpolate(
                recorte.unsqueeze(0),
                size=(alto, ancho),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            nuevas_anot = dict(anotaciones)
            nuevas_anot["boxes"] = cajas_final
            nuevas_anot["labels"] = anotaciones["labels"][indices]
            nuevas_anot["area"] = dims[:, 0] * dims[:, 1]
            nuevas_anot["iscrowd"] = anotaciones["iscrowd"][indices]
            return redim, nuevas_anot

        return imagen, anotaciones


def construir_transformaciones(cfg_datos: dict) -> Callable | None:
    """Construye la composición de aumentos a partir de la sección ``datos`` del YAML."""
    cfg_aum = cfg_datos.get("aumentos")
    lista: list[Callable] = []

    if cfg_aum is None:
        prob_h = float(cfg_datos.get("volteo_horizontal", 0.0))
        if prob_h > 0:
            lista.append(VolteoHorizontalAleatorio(prob=prob_h))
    else:
        prob_h = float(cfg_aum.get("volteo_horizontal", 0.0))
        if prob_h > 0:
            lista.append(VolteoHorizontalAleatorio(prob=prob_h))

        prob_v = float(cfg_aum.get("volteo_vertical", 0.0))
        if prob_v > 0:
            lista.append(VolteoVerticalAleatorio(prob=prob_v))

        cfg_recorte = cfg_aum.get("recorte_aleatorio")
        if cfg_recorte and float(cfg_recorte.get("prob", 0.0)) > 0:
            lista.append(
                RecorteAleatorioConCajas(
                    escala_min=float(cfg_recorte.get("escala_min", 0.65)),
                    escala_max=float(cfg_recorte.get("escala_max", 0.95)),
                    visibilidad_min=float(cfg_recorte.get("visibilidad_min", 0.35)),
                    prob=float(cfg_recorte.get("prob", 0.5)),
                )
            )

        cfg_color = cfg_aum.get("color_jitter")
        if cfg_color and float(cfg_color.get("prob", 0.0)) > 0:
            lista.append(
                AjusteColorAleatorio(
                    brillo=float(cfg_color.get("brillo", 0.2)),
                    contraste=float(cfg_color.get("contraste", 0.2)),
                    saturacion=float(cfg_color.get("saturacion", 0.2)),
                    tono=float(cfg_color.get("tono", 0.04)),
                    prob=float(cfg_color.get("prob", 0.7)),
                )
            )

    if not lista:
        return None
    if len(lista) == 1:
        return lista[0]
    return ComponerTransformaciones(lista)
