"""Estrategias de post-procesado, Test-Time Augmentation (TTA) y Ensemble (WBF)."""

from __future__ import annotations

import numpy as np
import torch


def _iou_matriz(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """Calcula el IoU entre una caja ``(4,)`` y una matriz de cajas ``(M, 4)``."""
    ix1 = np.maximum(caja[0], cajas[:, 0])
    iy1 = np.maximum(caja[1], cajas[:, 1])
    ix2 = np.minimum(caja[2], cajas[:, 2])
    iy2 = np.minimum(caja[3], cajas[:, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, caja[2] - caja[0]) * np.maximum(0.0, caja[3] - caja[1])
    area_b = np.maximum(0.0, cajas[:, 2] - cajas[:, 0]) * np.maximum(0.0, cajas[:, 3] - cajas[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def weighted_boxes_fusion(
    listas_cajas: list[np.ndarray],
    listas_scores: list[np.ndarray],
    listas_etiquetas: list[np.ndarray],
    pesos: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    max_detecciones: int = 100,
) -> dict[str, torch.Tensor]:
    """Fusiona detecciones de múltiples vistas (TTA) o modelos (Ensemble) mediante WBF.

    Implementación fiel al algoritmo Weighted Boxes Fusion (Solovyev et al., 2021):
    agrupa cajas solapadas de la misma clase y promedia sus coordenadas ponderadas
    por confianza y peso del modelo/vista, penalizando detecciones que solo aparecen
    en un subconjunto de los modelos.
    """
    n_modelos = len(listas_cajas)
    if n_modelos == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    if pesos is None:
        pesos_arr = np.ones(n_modelos, dtype=np.float64)
    else:
        pesos_arr = np.asarray(pesos, dtype=np.float64)
    suma_pesos = float(pesos_arr.sum())

    # Recopilar cajas por clase.
    por_clase: dict[int, list[tuple[float, float, np.ndarray]]] = {}
    for idx_m in range(n_modelos):
        cajas = np.asarray(listas_cajas[idx_m], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(listas_scores[idx_m], dtype=np.float64).reshape(-1)
        etiquetas = np.asarray(listas_etiquetas[idx_m], dtype=np.int64).reshape(-1)
        peso_m = float(pesos_arr[idx_m])

        for caja, score, etq in zip(cajas, scores, etiquetas):
            if score < skip_box_thr:
                continue
            if caja[2] <= caja[0] or caja[3] <= caja[1]:
                continue
            por_clase.setdefault(int(etq), []).append((float(score * peso_m), peso_m, caja))

    cajas_out: list[np.ndarray] = []
    scores_out: list[float] = []
    etiquetas_out: list[int] = []

    for etq, items in por_clase.items():
        # Ordenar de mayor a menor confianza ponderada.
        items.sort(key=lambda x: -x[0])

        clusters_cajas: list[list[np.ndarray]] = []
        clusters_scores: list[list[float]] = []
        cajas_fusionadas: list[np.ndarray] = []

        for score_p, _, caja in items:
            if not cajas_fusionadas:
                clusters_cajas.append([caja])
                clusters_scores.append([score_p])
                cajas_fusionadas.append(caja.copy())
                continue

            ious = _iou_matriz(caja, np.asarray(cajas_fusionadas))
            mejor_idx = int(np.argmax(ious))
            if ious[mejor_idx] >= iou_thr:
                clusters_cajas[mejor_idx].append(caja)
                clusters_scores[mejor_idx].append(score_p)

                w = np.asarray(clusters_scores[mejor_idx], dtype=np.float64)[:, None]
                b = np.asarray(clusters_cajas[mejor_idx], dtype=np.float64)
                cajas_fusionadas[mejor_idx] = (b * w).sum(axis=0) / max(w.sum(), 1e-12)
            else:
                clusters_cajas.append([caja])
                clusters_scores.append([score_p])
                cajas_fusionadas.append(caja.copy())

        for caja_f, lista_sc in zip(cajas_fusionadas, clusters_scores):
            # Confianza WBF: promedio ponderado escalado por cobertura de modelos.
            score_final = float(sum(lista_sc) / suma_pesos)
            score_final = min(1.0, score_final)
            cajas_out.append(caja_f)
            scores_out.append(score_final)
            etiquetas_out.append(etq)

    if not cajas_out:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    orden = np.argsort(-np.asarray(scores_out))[:max_detecciones]
    return {
        "boxes": torch.tensor(np.asarray(cajas_out)[orden], dtype=torch.float32),
        "scores": torch.tensor(np.asarray(scores_out)[orden], dtype=torch.float32),
        "labels": torch.tensor(np.asarray(etiquetas_out)[orden], dtype=torch.int64),
    }


@torch.no_grad()
def predecir_imagen_tta(
    modelo: torch.nn.Module,
    imagen: torch.Tensor,
    dispositivo: torch.device,
    modos: tuple[str, ...] = ("original", "hflip"),
    iou_wbf: float = 0.55,
    skip_box_thr: float = 0.05,
    usar_amp: bool = True,
) -> dict[str, torch.Tensor]:
    """Ejecuta Test-Time Augmentation sobre una imagen y fusiona con WBF.

    Modos soportados: ``"original"``, ``"hflip"``, ``"vflip"``, ``"hvflip"``.
    """
    _, alto, ancho = imagen.shape
    amp_activo = usar_amp and dispositivo.type == "cuda"

    vistas: list[torch.Tensor] = []
    for modo in modos:
        if modo == "original":
            vistas.append(imagen)
        elif modo == "hflip":
            vistas.append(torch.flip(imagen, dims=[2]))
        elif modo == "vflip":
            vistas.append(torch.flip(imagen, dims=[1]))
        elif modo == "hvflip":
            vistas.append(torch.flip(imagen, dims=[1, 2]))
        else:
            raise ValueError(f"Modo TTA no soportado: {modo!r}")

    salidas = []
    with torch.amp.autocast(device_type=dispositivo.type, enabled=amp_activo):
        for v in vistas:
            salidas.append(modelo([v.to(dispositivo)])[0])

    listas_cajas: list[np.ndarray] = []
    listas_scores: list[np.ndarray] = []
    listas_etiquetas: list[np.ndarray] = []

    for modo, salida in zip(modos, salidas):
        cajas = salida["boxes"].detach().cpu().float().numpy().copy()
        scores = salida["scores"].detach().cpu().float().numpy()
        etiquetas = salida["labels"].detach().cpu().numpy()

        if len(cajas) > 0:
            if modo in ("hflip", "hvflip"):
                cajas[:, [0, 2]] = float(ancho) - cajas[:, [2, 0]]
            if modo in ("vflip", "hvflip"):
                cajas[:, [1, 3]] = float(alto) - cajas[:, [3, 1]]

        listas_cajas.append(cajas)
        listas_scores.append(scores)
        listas_etiquetas.append(etiquetas)

    return weighted_boxes_fusion(
        listas_cajas=listas_cajas,
        listas_scores=listas_scores,
        listas_etiquetas=listas_etiquetas,
        iou_thr=iou_wbf,
        skip_box_thr=skip_box_thr,
    )


def combinar_predicciones_ensemble(
    predicciones_por_modelo: list[list[dict[str, torch.Tensor]]],
    pesos: list[float] | None = None,
    iou_wbf: float = 0.55,
    skip_box_thr: float = 0.05,
    max_detecciones: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Combina las predicciones de varios modelos imagen a imagen mediante WBF."""
    if not predicciones_por_modelo:
        raise ValueError("Se necesita al menos un conjunto de predicciones.")
    n_imagenes = len(predicciones_por_modelo[0])
    for preds in predicciones_por_modelo:
        if len(preds) != n_imagenes:
            raise ValueError("Todos los modelos deben tener el mismo número de imágenes.")

    fusionadas: list[dict[str, torch.Tensor]] = []
    for idx in range(n_imagenes):
        listas_cajas = [m[idx]["boxes"].cpu().numpy() for m in predicciones_por_modelo]
        listas_scores = [m[idx]["scores"].cpu().numpy() for m in predicciones_por_modelo]
        listas_etiquetas = [m[idx]["labels"].cpu().numpy() for m in predicciones_por_modelo]
        fusionadas.append(
            weighted_boxes_fusion(
                listas_cajas=listas_cajas,
                listas_scores=listas_scores,
                listas_etiquetas=listas_etiquetas,
                pesos=pesos,
                iou_thr=iou_wbf,
                skip_box_thr=skip_box_thr,
                max_detecciones=max_detecciones,
            )
        )
    return fusionadas
