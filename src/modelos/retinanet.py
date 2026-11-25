"""Adaptación de RetinaNet ResNet50-FPN v2 (torchvision) para Corrosion-bi3q3."""

from __future__ import annotations

import math

from torch import nn
from torchvision.models.detection import (
    RetinaNet,
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator

from src.modelos.faster_rcnn import NUM_CLASES_DEFECTO, establecer_capas_entrenables_backbone
from src.perdidas import aplicar_perdida_retinanet


def crear_retinanet(
    num_clases: int = NUM_CLASES_DEFECTO,
    preentrenado: bool = True,
    capas_entrenables_backbone: int | None = None,
    tamano_min: int = 800,
    tamano_max: int = 1333,
    tamanos_anchor: list[list[int]] | None = None,
    relaciones_aspecto: list[float] | None = None,
    perdida: dict | None = None,
    **kwargs,
) -> RetinaNet:
    """Crea RetinaNet v2 con pesos COCO y geometría de anchors opcionales.

    La pérdida focal de torchvision usa las etiquetas como índices directos.
    Se reserva el índice 0 y se conservan las etiquetas 1, 2 y 3 del dataset.
    Si se omiten las capas entrenables, torchvision usa 3 con COCO y 5 sin pesos.
    """
    pesos = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if preentrenado else None
    modelo = retinanet_resnet50_fpn_v2(
        weights=pesos,
        weights_backbone=None,
        num_classes=None if preentrenado else num_clases,
        trainable_backbone_layers=capas_entrenables_backbone if preentrenado else None,
        min_size=tamano_min,
        max_size=tamano_max,
        **kwargs,
    )

    cabeza = modelo.head.classification_head
    num_anchors = cabeza.num_anchors
    if tamanos_anchor is not None or relaciones_aspecto is not None:
        generador_original = modelo.anchor_generator
        tamanos = tuple(tuple(nivel) for nivel in tamanos_anchor) if tamanos_anchor is not None else generador_original.sizes
        ratios = (tuple(relaciones_aspecto),) * len(tamanos) if relaciones_aspecto is not None else generador_original.aspect_ratios
        if len(tamanos) != len(generador_original.sizes):
            raise ValueError("RetinaNet requiere un conjunto de tamaños por nivel de la FPN.")
        generador = AnchorGenerator(tamanos, ratios)
        cantidades = generador.num_anchors_per_location()
        if not cantidades[0] or len(set(cantidades)) != 1:
            raise ValueError("Todos los niveles deben tener el mismo número positivo de anchors.")
        num_anchors = cantidades[0]
        modelo.anchor_generator = generador

    # Cargar primero COCO permite conservar las torres de la cabeza y su GroupNorm.
    if preentrenado or num_anchors != cabeza.num_anchors:
        logits = cabeza.cls_logits
        cabeza.cls_logits = nn.Conv2d(
            logits.in_channels,
            num_anchors * num_clases,
            kernel_size=logits.kernel_size,
            stride=logits.stride,
            padding=logits.padding,
        )
        cabeza.num_classes = num_clases
        # Inicialización de torchvision con una probabilidad inicial de 0.01 por clase.
        nn.init.normal_(cabeza.cls_logits.weight, std=0.01)
        nn.init.constant_(cabeza.cls_logits.bias, -math.log(99))
    if num_anchors != cabeza.num_anchors:
        regresion = modelo.head.regression_head
        caja = regresion.bbox_reg
        regresion.bbox_reg = nn.Conv2d(
            caja.in_channels, num_anchors * 4, caja.kernel_size, stride=caja.stride, padding=caja.padding,
        )
        nn.init.normal_(regresion.bbox_reg.weight, std=0.01)
        nn.init.zeros_(regresion.bbox_reg.bias)
        cabeza.num_anchors = num_anchors
    if not preentrenado and capas_entrenables_backbone is not None:
        # torchvision ignora el número de capas entrenables si se construye sin pesos.
        establecer_capas_entrenables_backbone(modelo, capas_entrenables_backbone)
    if perdida:
        aplicar_perdida_retinanet(modelo, perdida)

    return modelo
