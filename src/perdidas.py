"""Funciones de pérdida personalizadas para Faster R-CNN v2 y RetinaNet v2.

Permite experimentar con:
- Pérdida de clasificación ponderada por clase (Weighted Cross-Entropy) o Focal Loss
  para paliar el desequilibrio de clases en Corrosion-bi3q3 (Slippage < corrosion < crack).
- Pérdida de regresión de cajas basada en IoU (GIoU / CIoU) en lugar de Smooth L1,
  que optimiza directamente la superposición geométrica de cajas pequeñas y alargadas.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss, generalized_box_iou_loss, sigmoid_focal_loss


def _perdida_clasificacion(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    tipo_cls: str,
    pesos_clases: list[float] | None,
    focal_gamma: float,
) -> torch.Tensor:
    peso_tensor = None
    if pesos_clases is not None:
        peso_tensor = torch.tensor(
            pesos_clases,
            dtype=class_logits.dtype,
            device=class_logits.device,
        )

    if tipo_cls == "ce":
        return F.cross_entropy(class_logits, labels, weight=peso_tensor)

    if tipo_cls == "focal":
        ce = F.cross_entropy(class_logits, labels, weight=peso_tensor, reduction="none")
        pt = torch.exp(-F.cross_entropy(class_logits, labels, reduction="none"))
        perdida = ((1.0 - pt) ** focal_gamma) * ce
        return perdida.mean()

    raise ValueError(f"Tipo de pérdida de clasificación desconocido: {tipo_cls!r}")


def _perdida_regresion_cajas(
    box_regression: torch.Tensor,
    labels: torch.Tensor,
    regression_targets: torch.Tensor,
    proposals_cat: torch.Tensor,
    box_coder,
    tipo_caja: str,
    peso_caja: float,
    smooth_l1_beta: float,
) -> torch.Tensor:
    pos_idx = torch.where(labels > 0)[0]
    if pos_idx.numel() == 0:
        return box_regression.sum() * 0.0

    labels_pos = labels[pos_idx]
    n_rois = labels.shape[0]
    box_regression = box_regression.reshape(n_rois, box_regression.size(-1) // 4, 4)

    deltas_pred = box_regression[pos_idx, labels_pos]
    deltas_gt = regression_targets[pos_idx]

    if tipo_caja == "smooth_l1":
        perdida = F.smooth_l1_loss(
            deltas_pred,
            deltas_gt,
            beta=smooth_l1_beta,
            reduction="sum",
        )
        return peso_caja * perdida / labels.numel()

    props_pos = proposals_cat[pos_idx]
    cajas_pred = box_coder.decode_single(deltas_pred, props_pos)
    cajas_gt = box_coder.decode_single(deltas_gt, props_pos)

    # Garantizar cajas válidas (x2 >= x1, y2 >= y1) para estabilidad numérica de IoU.
    cajas_pred = torch.stack(
        [
            torch.minimum(cajas_pred[:, 0], cajas_pred[:, 2]),
            torch.minimum(cajas_pred[:, 1], cajas_pred[:, 3]),
            torch.maximum(cajas_pred[:, 0], cajas_pred[:, 2]),
            torch.maximum(cajas_pred[:, 1], cajas_pred[:, 3]),
        ],
        dim=1,
    )

    if tipo_caja == "giou":
        perdida = generalized_box_iou_loss(cajas_pred, cajas_gt, reduction="sum")
    elif tipo_caja == "ciou":
        perdida = complete_box_iou_loss(cajas_pred, cajas_gt, reduction="sum")
    else:
        raise ValueError(f"Tipo de pérdida de caja desconocido: {tipo_caja!r}")

    return peso_caja * perdida / labels.numel()


def aplicar_perdida_personalizada(modelo: torch.nn.Module, cfg_perdida: dict) -> None:
    """Sustituye el cálculo de pérdidas en ``modelo.roi_heads.forward`` en modo entrenamiento.

    En modo evaluación (``not self.training``) delega al comportamiento original.
    """
    tipo_cls = str(cfg_perdida.get("clasificacion", "ce")).strip().lower()
    pesos_clases = cfg_perdida.get("pesos_clases", None)
    focal_gamma = float(cfg_perdida.get("focal_gamma", 2.0))

    tipo_caja = str(cfg_perdida.get("regresion_caja", "smooth_l1")).strip().lower()
    peso_caja = float(cfg_perdida.get("peso_caja", 1.0 if tipo_caja == "smooth_l1" else 2.5))
    smooth_l1_beta = float(cfg_perdida.get("smooth_l1_beta", 1.0 / 9.0))

    roi_heads = modelo.roi_heads
    forward_original = roi_heads.forward

    def forward_personalizado(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return forward_original(features, proposals, image_shapes, targets)

        if targets is None:
            raise ValueError("Se requieren targets durante el entrenamiento.")

        proposals_sel, _, labels, regression_targets = self.select_training_samples(
            proposals, targets
        )
        box_features = self.box_roi_pool(features, proposals_sel, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        labels_cat = torch.cat(labels, dim=0)
        reg_targets_cat = torch.cat(regression_targets, dim=0)
        proposals_cat = torch.cat(proposals_sel, dim=0)

        loss_cls = _perdida_clasificacion(
            class_logits=class_logits,
            labels=labels_cat,
            tipo_cls=tipo_cls,
            pesos_clases=pesos_clases,
            focal_gamma=focal_gamma,
        )
        loss_box = _perdida_regresion_cajas(
            box_regression=box_regression,
            labels=labels_cat,
            regression_targets=reg_targets_cat,
            proposals_cat=proposals_cat,
            box_coder=self.box_coder,
            tipo_caja=tipo_caja,
            peso_caja=peso_caja,
            smooth_l1_beta=smooth_l1_beta,
        )

        return [], {"loss_classifier": loss_cls, "loss_box_reg": loss_box}

    roi_heads.forward = MethodType(forward_personalizado, roi_heads)


def aplicar_perdida_retinanet(modelo: torch.nn.Module, cfg_perdida: dict) -> None:
    """Pondera los positivos de la pérdida focal y permite cambiar la regresión."""
    if cfg_perdida.get("clasificacion", "focal") != "focal":
        raise ValueError("RetinaNet requiere clasificación focal sigmoide.")

    clasificacion = modelo.head.classification_head
    pesos = cfg_perdida.get("pesos_clases")
    if pesos is not None:
        if len(pesos) != clasificacion.num_classes:
            raise ValueError("Se requiere un peso por índice de clase de RetinaNet.")

        def focal_ponderada(self, targets, head_outputs, matched_idxs):
            pesos_clase = head_outputs["cls_logits"].new_tensor(pesos)
            perdidas = []
            for objetivo, logits, coincidencias in zip(targets, head_outputs["cls_logits"], matched_idxs):
                positivos = coincidencias >= 0
                etiquetas = objetivo["labels"][coincidencias[positivos]]
                clases = torch.zeros_like(logits)
                clases[positivos, etiquetas] = 1
                validos = coincidencias != self.BETWEEN_THRESHOLDS
                clases_validas = clases[validos]
                focal = sigmoid_focal_loss(logits[validos], clases_validas, reduction="none")
                pesos_entrada = torch.where(clases_validas.bool(), pesos_clase, 1.0)
                perdidas.append((focal * pesos_entrada).sum() / positivos.sum().clamp_min(1))
            return torch.stack(perdidas).mean()

        clasificacion.compute_loss = MethodType(focal_ponderada, clasificacion)

    regresion = modelo.head.regression_head
    tipo_caja = cfg_perdida.get("regresion_caja", regresion._loss_type)
    if tipo_caja not in {"l1", "smooth_l1", "giou", "diou", "ciou"}:
        raise ValueError(f"Tipo de pérdida de caja desconocido: {tipo_caja!r}")
    regresion._loss_type = tipo_caja
    peso_caja = float(cfg_perdida.get("peso_caja", 1.0))
    if peso_caja != 1.0:
        calcular_regresion = regresion.compute_loss

        def regresion_ponderada(targets, head_outputs, anchors, matched_idxs):
            return peso_caja * calcular_regresion(targets, head_outputs, anchors, matched_idxs)

        regresion.compute_loss = regresion_ponderada
