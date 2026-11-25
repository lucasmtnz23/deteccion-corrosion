"""Evaluación completa de detectores sobre Corrosion-bi3q3.

Calcula:
- Métricas oficiales COCO (mAP@[0.50:0.95], mAP@0.50, mAP@0.75, por tamaño y por clase).
- Curvas Precisión-Recall (PR) por clase a IoU=0.50 y IoU=0.75.
- Tiempos de inferencia (ms/imagen y FPS) con sincronización CUDA.
- Desglose de errores estilo TIDE (TP, Loc, Cls, Dup, Bkg, Miss).
- Exportación/carga de predicciones en JSON para visualización y ensembles.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.dataset import CorrosionDataset, NOMBRES_CLASES, PARTICIONES, collate_fn
from src.modelos import crear_modelo


def _a_float(valor: torch.Tensor) -> float:
    return float(valor.detach().cpu().item())


def _iou_entre_cajas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz IoU ``(N, M)`` entre conjuntos de cajas ``xyxy``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def calcular_curvas_pr(
    predicciones: list[dict[str, torch.Tensor]],
    objetivos: list[dict[str, torch.Tensor]],
    iou_thr: float = 0.50,
    num_puntos: int = 101,
) -> dict[str, dict]:
    """Calcula curvas Precisión-Recall y AP (interpolación COCO 101 puntos) por clase."""
    curvas: dict[str, dict] = {}
    rejilla_recall = np.linspace(0.0, 1.0, num_puntos)

    for id_clase, nombre_clase in NOMBRES_CLASES.items():
        n_gt = 0
        gt_por_imagen: dict[int, np.ndarray] = {}
        preds_clase: list[tuple[float, int, np.ndarray]] = []

        for idx_img, (pred, obj) in enumerate(zip(predicciones, objetivos)):
            cajas_gt = obj["boxes"].cpu().numpy()
            etq_gt = obj["labels"].cpu().numpy()
            cajas_gt_c = cajas_gt[etq_gt == id_clase]
            gt_por_imagen[idx_img] = cajas_gt_c
            n_gt += len(cajas_gt_c)

            cajas_p = pred["boxes"].cpu().numpy()
            scores_p = pred["scores"].cpu().numpy()
            etq_p = pred["labels"].cpu().numpy()
            mascara = etq_p == id_clase
            for caja, score in zip(cajas_p[mascara], scores_p[mascara]):
                preds_clase.append((float(score), idx_img, caja))

        if n_gt == 0:
            curvas[nombre_clase] = {
                "n_gt": 0,
                "n_preds": len(preds_clase),
                "ap": -1.0,
                "recall": [],
                "precision": [],
                "precision_interp": [],
            }
            continue

        preds_clase.sort(key=lambda x: -x[0])
        emparejados = {idx: np.zeros(len(cajas), dtype=bool) for idx, cajas in gt_por_imagen.items()}
        tp = np.zeros(len(preds_clase), dtype=np.float64)
        fp = np.zeros(len(preds_clase), dtype=np.float64)

        for i, (_, idx_img, caja_p) in enumerate(preds_clase):
            cajas_gt_c = gt_por_imagen[idx_img]
            if len(cajas_gt_c) == 0:
                fp[i] = 1.0
                continue
            ious = _iou_entre_cajas(caja_p[None, :], cajas_gt_c)[0]

            # Buscar GT válidos que todavía no hayan sido emparejados.
            disponibles = np.where(
                (ious >= iou_thr) & (~emparejados[idx_img])
            )[0]

            if len(disponibles) > 0:
                # Elegir la caja disponible con mayor IoU.
                mejor_j = disponibles[np.argmax(ious[disponibles])]
                tp[i] = 1.0
                emparejados[idx_img][mejor_j] = True
            else:
                fp[i] = 1.0

        if len(preds_clase) == 0:
            curvas[nombre_clase] = {
                "n_gt": int(n_gt),
                "n_preds": 0,
                "ap": 0.0,
                "recall": rejilla_recall.tolist(),
                "precision": [0.0] * num_puntos,
                "precision_interp": [0.0] * num_puntos,
            }
            continue

        tp_acum = np.cumsum(tp)
        fp_acum = np.cumsum(fp)
        recall = tp_acum / float(n_gt)
        precision = tp_acum / np.maximum(tp_acum + fp_acum, 1e-12)
        env = np.maximum.accumulate(precision[::-1])[::-1]

        prec_101 = np.array(
            [env[recall >= r][0] if np.any(recall >= r) else 0.0 for r in rejilla_recall],
            dtype=np.float64,
        )
        ap = float(np.mean(prec_101))

        curvas[nombre_clase] = {
            "n_gt": int(n_gt),
            "n_preds": int(len(preds_clase)),
            "ap": ap,
            "recall": rejilla_recall.tolist(),
            "precision_interp": prec_101.tolist(),
        }

    return curvas


def analizar_errores_deteccion(
    predicciones: list[dict[str, torch.Tensor]],
    objetivos: list[dict[str, torch.Tensor]],
    score_thr: float = 0.50,
    iou_tp: float = 0.50,
    iou_bkg: float = 0.10,
) -> dict[str, int | float]:
    """Desglosa detecciones y fallos en categorías estilo TIDE a un umbral de confianza.

    Categorías:
    - ``TP``: clase correcta e ``IoU >= iou_tp`` con GT no emparejado.
    - ``Dup``: clase correcta e ``IoU >= iou_tp`` con GT ya emparejado.
    - ``Loc``: clase correcta y ``iou_bkg <= IoU < iou_tp``.
    - ``Cls``: clase incorrecta e ``IoU >= iou_tp`` con algún GT de otra clase.
    - ``Both``: clase incorrecta y ``iou_bkg <= IoU < iou_tp``.
    - ``Bkg``: ``IoU < iou_bkg`` con cualquier GT (falso positivo de fondo).
    - ``Miss``: objetos GT no detectados (falsos negativos).
    """
    contadores = {
        "TP": 0,
        "Dup": 0,
        "Loc": 0,
        "Cls": 0,
        "Both": 0,
        "Bkg": 0,
        "Miss": 0,
        "total_gt": 0,
        "total_preds": 0,
    }

    for pred, obj in zip(predicciones, objetivos):
        cajas_gt = obj["boxes"].cpu().numpy()
        etq_gt = obj["labels"].cpu().numpy()
        contadores["total_gt"] += len(cajas_gt)
        usado = np.zeros(len(cajas_gt), dtype=bool)

        cajas_p = pred["boxes"].cpu().numpy()
        scores_p = pred["scores"].cpu().numpy()
        etq_p = pred["labels"].cpu().numpy()

        mascara = scores_p >= score_thr
        cajas_p = cajas_p[mascara]
        scores_p = scores_p[mascara]
        etq_p = etq_p[mascara]
        orden = np.argsort(-scores_p)
        contadores["total_preds"] += len(orden)

        for idx in orden:
            caja = cajas_p[idx]
            etq = int(etq_p[idx])
            if len(cajas_gt) == 0:
                contadores["Bkg"] += 1
                continue

            ious = _iou_entre_cajas(caja[None, :], cajas_gt)[0]
            misma_clase = etq_gt == etq

            mejor_iou_misma = -1.0
            mejor_j_libre = -1
            mejor_iou_libre = -1.0
            if np.any(misma_clase):
                indices_misma = np.where(misma_clase)[0]
                mejor_iou_misma = float(np.max(ious[indices_misma]))
                libres = [j for j in indices_misma if not usado[j]]
                if libres:
                    j_best = max(libres, key=lambda j: ious[j])
                    mejor_iou_libre = float(ious[j_best])
                    mejor_j_libre = int(j_best)

            if mejor_iou_libre >= iou_tp:
                contadores["TP"] += 1
                usado[mejor_j_libre] = True
            elif mejor_iou_misma >= iou_tp:
                contadores["Dup"] += 1
            elif mejor_iou_misma >= iou_bkg:
                contadores["Loc"] += 1
            else:
                mejor_iou_global = float(np.max(ious))
                if mejor_iou_global >= iou_tp:
                    contadores["Cls"] += 1
                elif mejor_iou_global >= iou_bkg:
                    contadores["Both"] += 1
                else:
                    contadores["Bkg"] += 1

        contadores["Miss"] += int(np.sum(~usado))

    tp = contadores["TP"]
    total_p = contadores["total_preds"]
    total_gt = contadores["total_gt"]
    precision = tp / total_p if total_p > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "score_thr": score_thr,
        **contadores,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluar_predicciones(
    predicciones: list[dict[str, torch.Tensor]],
    objetivos: list[dict[str, torch.Tensor]],
    incluir_pr: bool = False,
    incluir_errores: bool = False,
) -> dict:
    """Calcula COCO mAP (y opcionalmente curvas PR y desglose TIDE) desde predicciones."""
    metrica = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )
    metrica.update(predicciones, objetivos)
    bruto = metrica.compute()

    resumen: dict = {
        "map": _a_float(bruto["map"]),
        "map_50": _a_float(bruto["map_50"]),
        "map_75": _a_float(bruto["map_75"]),
        "map_small": _a_float(bruto["map_small"]),
        "map_medium": _a_float(bruto["map_medium"]),
        "map_large": _a_float(bruto["map_large"]),
        "mar_100": _a_float(bruto["mar_100"]),
    }

    ap_por_clase: dict[str, float] = {}
    clases = bruto.get("classes")
    map_por_clase = bruto.get("map_per_class")
    if clases is not None and map_por_clase is not None:
        clases_lista = clases.reshape(-1).tolist()
        valores_lista = map_por_clase.reshape(-1).tolist()
        for id_clase, valor_ap in zip(clases_lista, valores_lista):
            nombre = NOMBRES_CLASES.get(int(id_clase), str(id_clase))
            ap_por_clase[nombre] = float(valor_ap)
    resumen["map_por_clase"] = ap_por_clase

    if incluir_pr:
        resumen["curvas_pr_50"] = calcular_curvas_pr(predicciones, objetivos, iou_thr=0.50)
        resumen["curvas_pr_75"] = calcular_curvas_pr(predicciones, objetivos, iou_thr=0.75)

    if incluir_errores:
        resumen["analisis_errores_05"] = analizar_errores_deteccion(
            predicciones, objetivos, score_thr=0.50
        )
        resumen["analisis_errores_03"] = analizar_errores_deteccion(
            predicciones, objetivos, score_thr=0.30
        )

    return resumen


@torch.no_grad()
def inferir_dataset(
    modelo: torch.nn.Module,
    cargador: DataLoader,
    dispositivo: torch.device,
    usar_amp: bool = True,
    mostrar_progreso: bool = True,
    medir_tiempo: bool = False,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], dict[str, float]]:
    """Ejecuta inferencia sobre un DataLoader devolviendo predicciones, objetivos y tiempos."""
    modelo.eval()
    amp_activo = usar_amp and dispositivo.type == "cuda"

    predicciones: list[dict[str, torch.Tensor]] = []
    objetivos: list[dict[str, torch.Tensor]] = []

    tiempo_inferencia_s = 0.0
    imagenes_medidas = 0
    lote_warmup = medir_tiempo

    iterador = tqdm(cargador, desc="Evaluando", leave=False) if mostrar_progreso else cargador

    for imagenes, anotaciones in iterador:
        imagenes_gpu = [img.to(dispositivo) for img in imagenes]

        if medir_tiempo and dispositivo.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast(device_type=dispositivo.type, enabled=amp_activo):
            salidas = modelo(imagenes_gpu)

        if medir_tiempo and dispositivo.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if medir_tiempo:
            if lote_warmup:
                lote_warmup = False
            else:
                tiempo_inferencia_s += t1 - t0
                imagenes_medidas += len(imagenes)

        for salida in salidas:
            predicciones.append(
                {
                    "boxes": salida["boxes"].detach().cpu().float(),
                    "scores": salida["scores"].detach().cpu().float(),
                    "labels": salida["labels"].detach().cpu(),
                }
            )
        for anot in anotaciones:
            objetivos.append(
                {
                    "boxes": anot["boxes"].detach().cpu().float(),
                    "labels": anot["labels"].detach().cpu(),
                    "image_id": anot["image_id"].detach().cpu(),
                }
            )

    tiempos: dict[str, float] = {}
    if medir_tiempo and imagenes_medidas > 0:
        ms_por_imagen = (tiempo_inferencia_s / imagenes_medidas) * 1000.0
        tiempos = {
            "ms_por_imagen": round(ms_por_imagen, 2),
            "fps": round(imagenes_medidas / max(tiempo_inferencia_s, 1e-9), 2),
            "imagenes_medidas": imagenes_medidas,
        }

    return predicciones, objetivos, tiempos


@torch.no_grad()
def evaluar_modelo(
    modelo: torch.nn.Module,
    cargador: DataLoader,
    dispositivo: torch.device,
    usar_amp: bool = True,
    mostrar_progreso: bool = True,
    detallado: bool = False,
    devolver_predicciones: bool = False,
):
    """Evalúa un detector en un DataLoader y devuelve métricas COCO (y extras si ``detallado``)."""
    predicciones, objetivos, tiempos = inferir_dataset(
        modelo=modelo,
        cargador=cargador,
        dispositivo=dispositivo,
        usar_amp=usar_amp,
        mostrar_progreso=mostrar_progreso,
        medir_tiempo=detallado,
    )
    resumen = evaluar_predicciones(
        predicciones=predicciones,
        objetivos=objetivos,
        incluir_pr=detallado,
        incluir_errores=detallado,
    )
    if tiempos:
        resumen["tiempos"] = tiempos

    if devolver_predicciones:
        return resumen, predicciones, objetivos
    return resumen


def guardar_predicciones(
    ruta: Path,
    predicciones: list[dict[str, torch.Tensor]],
    objetivos: list[dict[str, torch.Tensor]],
) -> None:
    """Guarda predicciones (y ``image_id``) en formato JSON reproducible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    registros = []
    for pred, obj in zip(predicciones, objetivos):
        registros.append(
            {
                "image_id": int(obj["image_id"].item()),
                "boxes": pred["boxes"].cpu().tolist(),
                "scores": pred["scores"].cpu().tolist(),
                "labels": pred["labels"].cpu().tolist(),
            }
        )
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(registros, archivo)


def _predicciones_a_tensores(registros: list[dict]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(r["scores"], dtype=torch.float32).reshape(-1),
            "labels": torch.tensor(r["labels"], dtype=torch.int64).reshape(-1),
        }
        for r in registros
    ]


def cargar_predicciones(ruta: Path) -> list[dict[str, torch.Tensor]]:
    """Carga predicciones previamente guardadas en JSON."""
    with open(ruta, encoding="utf-8") as archivo:
        return _predicciones_a_tensores(json.load(archivo))


def cargar_predicciones_alineadas(
    ruta: Path,
    objetivos: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Valida los image_id del JSON y devuelve las predicciones en orden de anotación."""
    with ruta.open(encoding="utf-8") as archivo:
        registros = json.load(archivo)
    if not isinstance(registros, list):
        raise ValueError(f"{ruta}: se esperaba una lista de registros de predicción.")

    por_id = {}
    duplicados = set()
    for numero, registro in enumerate(registros, start=1):
        id_imagen = registro.get("image_id") if isinstance(registro, dict) else None
        if type(id_imagen) is not int:
            raise ValueError(f"{ruta}: `image_id` ausente o no entero en el registro {numero}.")
        if id_imagen in por_id:
            duplicados.add(id_imagen)
        por_id[id_imagen] = registro

    ids_objetivos = [int(objetivo["image_id"].item()) for objetivo in objetivos]
    esperados = set(ids_objetivos)
    if len(esperados) != len(ids_objetivos):
        raise ValueError(f"{ruta}: las anotaciones de la partición tienen `image_id` duplicados.")
    ausentes = esperados - por_id.keys()
    inesperados = por_id.keys() - esperados
    errores = [
        f"{tipo}: {sorted(ids)}"
        for tipo, ids in (
            ("duplicados", duplicados), ("ausentes", ausentes), ("inesperados", inesperados),
        )
        if ids
    ]
    if errores:
        raise ValueError(f"{ruta}: image_id " + ", ".join(errores) + ".")
    return _predicciones_a_tensores([por_id[id_imagen] for id_imagen in ids_objetivos])


def formatear_metricas(metricas: dict) -> str:
    """Devuelve un resumen legible de las métricas COCO y de latencia."""
    lineas = [
        f"mAP@[0.50:0.95]: {metricas['map']:.4f}",
        f"mAP@0.50:        {metricas['map_50']:.4f}",
        f"mAP@0.75:        {metricas['map_75']:.4f}",
        f"mAP_small:       {metricas['map_small']:.4f}",
        f"mAP_medium:      {metricas['map_medium']:.4f}",
        f"mAP_large:       {metricas['map_large']:.4f}",
        f"mAR@100:         {metricas['mar_100']:.4f}",
    ]
    por_clase = metricas.get("map_por_clase", {})
    if isinstance(por_clase, dict) and por_clase:
        lineas.append("AP por clase (mAP@[0.50:0.95]):")
        for nombre, valor in por_clase.items():
            etiqueta_valor = f"{valor:.4f}" if valor >= 0 else "N/A (sin ejemplos)"
            lineas.append(f"  - {nombre}: {etiqueta_valor}")

    curvas_50 = metricas.get("curvas_pr_50", {})
    if isinstance(curvas_50, dict) and curvas_50:
        lineas.append("AP@0.50 por clase:")
        for nombre, datos_c in curvas_50.items():
            ap = datos_c["ap"]
            etiqueta_valor = f"{ap:.4f} (GT={datos_c['n_gt']})" if ap >= 0 else "N/A"
            lineas.append(f"  - {nombre}: {etiqueta_valor}")

    tiempos = metricas.get("tiempos")
    if isinstance(tiempos, dict) and tiempos:
        lineas.append(
            f"Inferencia:      {tiempos['ms_por_imagen']:.2f} ms/img ({tiempos['fps']:.2f} FPS)"
        )

    errores = metricas.get("analisis_errores_05")
    if isinstance(errores, dict) and errores:
        lineas.append(
            f"Errores (thr=0.50): TP={errores['TP']} | Loc={errores['Loc']} | "
            f"Cls={errores['Cls']} | Dup={errores['Dup']} | Bkg={errores['Bkg']} | "
            f"Miss={errores['Miss']} | F1={errores['f1']:.4f}"
        )
    return "\n".join(lineas)


def cargar_modelo_desde_checkpoint(
    ruta_checkpoint: Path,
    dispositivo: torch.device,
    ruta_config: Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruye el modelo y carga los pesos guardados en un checkpoint."""
    checkpoint = torch.load(ruta_checkpoint, map_location=dispositivo, weights_only=False)

    if ruta_config is not None:
        with open(ruta_config, encoding="utf-8") as archivo:
            config = yaml.safe_load(archivo)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError(
            "El checkpoint no contiene 'config'; indica --config para reconstruir el modelo."
        )

    cfg_modelo = dict(config["modelo"])
    nombre = cfg_modelo.pop("nombre")
    # En inferencia no hace falta descargar pesos COCO ni envolver pérdidas de entrenamiento.
    cfg_modelo["preentrenado"] = False
    cfg_modelo.pop("perdida", None)
    modelo = crear_modelo(nombre, **cfg_modelo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.to(dispositivo)
    return modelo, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa un detector entrenado en Corrosion-bi3q3.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ruta al archivo .pth.")
    parser.add_argument(
        "--particion",
        choices=PARTICIONES,
        default="test",
        help="Partición sobre la que evaluar (por defecto: test).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Archivo YAML opcional si el checkpoint no incluye configuración.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de lote para evaluación.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers del DataLoader.")
    parser.add_argument("--guardar", type=Path, default=None, help="Ruta JSON para las métricas.")
    parser.add_argument(
        "--guardar-preds",
        type=Path,
        default=None,
        help="Ruta JSON opcional para guardar las predicciones.",
    )
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    modelo, _ = cargar_modelo_desde_checkpoint(args.checkpoint, dispositivo, args.config)
    dataset = CorrosionDataset(particion=args.particion)
    cargador = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    metricas, predicciones, objetivos = evaluar_modelo(
        modelo,
        cargador,
        dispositivo,
        detallado=True,
        devolver_predicciones=True,
    )
    print(f"\n=== Resultados en {args.particion} ===")
    print(formatear_metricas(metricas))

    if args.guardar is not None:
        args.guardar.parent.mkdir(parents=True, exist_ok=True)
        with open(args.guardar, "w", encoding="utf-8") as archivo:
            json.dump(metricas, archivo, indent=2, ensure_ascii=False)
        print(f"\nMétricas guardadas en {args.guardar}")

    if args.guardar_preds is not None:
        guardar_predicciones(args.guardar_preds, predicciones, objetivos)
        print(f"Predicciones guardadas en {args.guardar_preds}")


if __name__ == "__main__":
    main()
