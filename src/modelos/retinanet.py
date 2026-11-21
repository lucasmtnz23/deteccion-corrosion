"""Adaptación de RetinaNet ResNet50-FPN v2 (torchvision) para Corrosion-bi3q3."""

from __future__ import annotations

import math

from torch import nn
from torchvision.models.detection import (
    RetinaNet,
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from src.modelos.faster_rcnn import NUM_CLASES_DEFECTO, establecer_capas_entrenables_backbone


def crear_retinanet(
    num_clases: int = NUM_CLASES_DEFECTO,
    preentrenado: bool = True,
    capas_entrenables_backbone: int | None = None,
    tamano_min: int = 800,
    tamano_max: int = 1333,
    **kwargs,
) -> RetinaNet:
    """Crea RetinaNet v2 con los anchors originales y pesos COCO opcionales.

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

    if preentrenado:
        # Tras cargar COCO, cambiar solo los logits conserva GroupNorm y los pesos de la torre v2.
        cabeza = modelo.head.classification_head
        logits = cabeza.cls_logits
        cabeza.cls_logits = nn.Conv2d(
            logits.in_channels,
            cabeza.num_anchors * num_clases,
            kernel_size=logits.kernel_size,
            stride=logits.stride,
            padding=logits.padding,
        )
        cabeza.num_classes = num_clases
        # Inicialización de torchvision con una probabilidad inicial de 0.01 por clase.
        nn.init.normal_(cabeza.cls_logits.weight, std=0.01)
        nn.init.constant_(cabeza.cls_logits.bias, -math.log(99))
    elif capas_entrenables_backbone is not None:
        # torchvision ignora el número de capas entrenables si se construye sin pesos.
        establecer_capas_entrenables_backbone(modelo, capas_entrenables_backbone)

    return modelo
