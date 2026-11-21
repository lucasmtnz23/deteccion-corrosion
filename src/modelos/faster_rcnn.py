"""Definición y adaptación de Faster R-CNN v2 (torchvision) para Corrosion-bi3q3."""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from src.dataset import NOMBRES_CLASES
from src.perdidas import aplicar_perdida_personalizada

# 0 se reserva para el fondo; 1..3 corresponden a las clases anotadas.
NUM_CLASES_DEFECTO = len(NOMBRES_CLASES) + 1

# Orden de bloques del backbone ResNet-50 en torchvision (de más profundo a más superficial).
_CAPAS_RESNET50 = ("layer4", "layer3", "layer2", "layer1", "conv1")


def crear_faster_rcnn(
    num_clases: int = NUM_CLASES_DEFECTO,
    preentrenado: bool = True,
    capas_entrenables_backbone: int | None = None,
    tamano_min: int = 800,
    tamano_max: int = 1333,
    tamanos_anchor: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    relaciones_aspecto: list[list[float]] | list[float] | tuple | None = None,
    perdida: dict | None = None,
    **kwargs,
) -> FasterRCNN:
    """Descarga y adapta Faster R-CNN ResNet50-FPN v2 preentrenado en MS COCO.

    Usa ``FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`` (46.7 AP en MS COCO) y
    sustituye la cabeza ``box_predictor`` por ``FastRCNNPredictor`` para las
    clases de Corrosion-bi3q3 (incluyendo el fondo).

    Parámetros
    ----------
    num_clases : int
        Número total de clases incluyendo el fondo (4 para Corrosion-bi3q3).
    preentrenado : bool
        Si es ``True``, carga los pesos oficiales ``COCO_V1`` de torchvision.
    capas_entrenables_backbone : int | None
        Bloques entrenables del backbone ResNet-50 (0..5). Con ``None``,
        torchvision usa su valor por defecto (3 si está preentrenado).
    tamano_min : int
        Lado mínimo en la transformación interna de Faster R-CNN.
    tamano_max : int
        Lado máximo en la transformación interna de Faster R-CNN.
    tamanos_anchor : secuencia de secuencias de int | None
        Tamaños de anchors para los 5 niveles de la FPN.
    relaciones_aspecto : secuencia de float o de secuencias de float | None
        Relaciones de aspecto por nivel de la FPN.
    perdida : dict | None
        Configuración opcional de pérdida personalizada (p. ej. CIoU/GIoU y/o
        clasificación ponderada / Focal Loss) en ``roi_heads``.
    **kwargs :
        Argumentos extra de ``fasterrcnn_resnet50_fpn_v2`` (``box_nms_thresh``,
        ``box_score_thresh``, ``box_detections_per_img``, etc.).
    """
    pesos = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if preentrenado else None

    args_modelo = {
        "weights": pesos,
        "min_size": tamano_min,
        "max_size": tamano_max,
        **kwargs,
    }
    if preentrenado and capas_entrenables_backbone is not None:
        args_modelo["trainable_backbone_layers"] = capas_entrenables_backbone

    modelo = fasterrcnn_resnet50_fpn_v2(**args_modelo)
    if capas_entrenables_backbone is not None:
        establecer_capas_entrenables_backbone(modelo, capas_entrenables_backbone)

    # Sustituir la cabeza de clasificación y regresión para nuestras clases.
    canales_entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(canales_entrada, num_clases)

    # Adaptación opcional de anchors (clave para objetos pequeños y grietas alargadas).
    if tamanos_anchor is not None or relaciones_aspecto is not None:
        tamanos = (
            tuple(tuple(int(s) for s in nivel) for nivel in tamanos_anchor)
            if tamanos_anchor is not None
            else modelo.rpn.anchor_generator.sizes
        )
        if relaciones_aspecto is None:
            relaciones = modelo.rpn.anchor_generator.aspect_ratios
        elif relaciones_aspecto and isinstance(relaciones_aspecto[0], (int, float)):
            relaciones = (tuple(float(r) for r in relaciones_aspecto),) * len(tamanos)
        else:
            relaciones = tuple(tuple(float(r) for r in nivel) for nivel in relaciones_aspecto)

        generador_anterior = modelo.rpn.anchor_generator
        nuevo_generador = AnchorGenerator(sizes=tamanos, aspect_ratios=relaciones)
        modelo.rpn.anchor_generator = nuevo_generador

        if nuevo_generador.num_anchors_per_location() != generador_anterior.num_anchors_per_location():
            modelo.rpn.head = RPNHead(
                modelo.backbone.out_channels,
                nuevo_generador.num_anchors_per_location()[0],
                conv_depth=2,
            )

    if perdida:
        aplicar_perdida_personalizada(modelo, perdida)

    return modelo


def establecer_capas_entrenables_backbone(modelo: FasterRCNN, num_capas: int) -> None:
    """Ajusta qué bloques del backbone ResNet-50 (0..5) están descongelados.

    Útil para implementar descongelado progresivo durante el entrenamiento.
    """
    if not 0 <= num_capas <= 5:
        raise ValueError(f"num_capas debe estar entre 0 y 5, recibido: {num_capas}")

    capas_activas = set(_CAPAS_RESNET50[:num_capas])
    if num_capas == 5:
        capas_activas.add("bn1")

    for nombre, parametro in modelo.backbone.body.named_parameters():
        bloque = nombre.split(".")[0]
        parametro.requires_grad = bloque in capas_activas


def grupos_parametros_diferenciados(
    modelo: FasterRCNN,
    lr: float,
    factor_lr_backbone: float = 1.0,
    weight_decay: float = 1e-4,
) -> list[dict]:
    """Crea grupos de parámetros con tasa de aprendizaje diferenciada para el backbone.

    Permite aplicar una tasa menor (``lr * factor_lr_backbone``) al extractor
    ResNet-50 preentrenado y la tasa completa ``lr`` a FPN, RPN y cabeza RoI.
    """
    if abs(factor_lr_backbone - 1.0) < 1e-9:
        return [
            {
                "params": [p for p in modelo.parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

    params_backbone = []
    params_cabeza = []
    for nombre, parametro in modelo.named_parameters():
        if not parametro.requires_grad:
            continue
        if nombre.startswith("backbone.body."):
            params_backbone.append(parametro)
        else:
            params_cabeza.append(parametro)

    grupos = []
    if params_backbone:
        grupos.append(
            {
                "params": params_backbone,
                "lr": lr * factor_lr_backbone,
                "weight_decay": weight_decay,
            }
        )
    if params_cabeza:
        grupos.append(
            {
                "params": params_cabeza,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return grupos


def contar_parametros(modelo: torch.nn.Module) -> tuple[int, int]:
    """Devuelve ``(parametros_totales, parametros_entrenables)``."""
    totales = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return totales, entrenables


if __name__ == "__main__":
    modelo_prueba = crear_faster_rcnn()
    totales, entrenables = contar_parametros(modelo_prueba)
    print(f"Faster R-CNN v2 (COCO_V1) cargado correctamente ({NUM_CLASES_DEFECTO} clases).")
    print(f"Parámetros totales: {totales:,} | Entrenables: {entrenables:,}")
