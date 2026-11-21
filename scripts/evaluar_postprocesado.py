"""Evaluación de estrategias de inferencia: ajuste de NMS/confianza, TTA y Ensemble WBF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CorrosionDataset, collate_fn
from src.evaluar import (
    cargar_modelo_desde_checkpoint,
    cargar_predicciones,
    evaluar_modelo,
    evaluar_predicciones,
    formatear_metricas,
    guardar_predicciones,
)
from src.postprocesado import combinar_predicciones_ensemble, predecir_imagen_tta


def ejecutar_barrido_nms(
    modelo: torch.nn.Module,
    dispositivo: torch.device,
    umbrales_nms: list[float],
    umbrales_score: list[float],
    batch_size: int = 4,
    usar_amp: bool = True,
) -> dict:
    """Busca los mejores ``box_nms_thresh`` y ``box_score_thresh`` sobre validación
    y evalúa la mejor combinación sobre test."""
    ds_val = CorrosionDataset("validation")
    cargador_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    nms_orig = modelo.roi_heads.nms_thresh
    score_orig = modelo.roi_heads.score_thresh

    resultados_val = []
    mejor_cfg = {"nms_thresh": nms_orig, "score_thresh": score_orig}
    mejor_map = -1.0

    for nms_thr in umbrales_nms:
        for score_thr in umbrales_score:
            modelo.roi_heads.nms_thresh = nms_thr
            modelo.roi_heads.score_thresh = score_thr
            met = evaluar_modelo(
                modelo,
                cargador_val,
                dispositivo,
                usar_amp=usar_amp,
                mostrar_progreso=False,
            )
            fila = {
                "nms_thresh": nms_thr,
                "score_thresh": score_thr,
                "val_map": met["map"],
                "val_map_50": met["map_50"],
                "val_map_small": met["map_small"],
            }
            resultados_val.append(fila)
            print(
                f"  [VAL] NMS={nms_thr:.2f}, score={score_thr:.2f} -> "
                f"mAP={met['map']:.4f}, mAP50={met['map_50']:.4f}"
            )
            if met["map"] > mejor_map:
                mejor_map = met["map"]
                mejor_cfg = {"nms_thresh": nms_thr, "score_thresh": score_thr}

    # Evaluar la mejor configuración en test.
    modelo.roi_heads.nms_thresh = mejor_cfg["nms_thresh"]
    modelo.roi_heads.score_thresh = mejor_cfg["score_thresh"]
    ds_test = CorrosionDataset("test")
    cargador_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    metricas_test, preds_test, objs_test = evaluar_modelo(
        modelo,
        cargador_test,
        dispositivo,
        usar_amp=usar_amp,
        detallado=True,
        devolver_predicciones=True,
    )

    # Restaurar valores originales del modelo.
    modelo.roi_heads.nms_thresh = nms_orig
    modelo.roi_heads.score_thresh = score_orig

    return {
        "mejor_config": mejor_cfg,
        "mejor_val_map": mejor_map,
        "barrido_val": resultados_val,
        "test": metricas_test,
        "_preds_test": preds_test,
        "_objs_test": objs_test,
    }


def ejecutar_tta(
    modelo: torch.nn.Module,
    dispositivo: torch.device,
    particion: str = "test",
    modos: tuple[str, ...] = ("original", "hflip"),
    iou_wbf: float = 0.55,
    skip_box_thr: float = 0.05,
    usar_amp: bool = True,
) -> tuple[dict, list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Evalúa Test-Time Augmentation con WBF midiendo mAP y latencia por imagen."""
    modelo.eval()
    ds = CorrosionDataset(particion)
    predicciones = []
    objetivos = []

    tiempo_s = 0.0
    imagenes_medidas = 0

    for idx in tqdm(range(len(ds)), desc=f"TTA ({'+'.join(modos)})"):
        imagen, anot = ds[idx]
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred = predecir_imagen_tta(
            modelo=modelo,
            imagen=imagen,
            dispositivo=dispositivo,
            modos=modos,
            iou_wbf=iou_wbf,
            skip_box_thr=skip_box_thr,
            usar_amp=usar_amp,
        )

        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if idx > 0:  # primera imagen como warmup
            tiempo_s += t1 - t0
            imagenes_medidas += 1

        predicciones.append(pred)
        objetivos.append(
            {
                "boxes": anot["boxes"].float(),
                "labels": anot["labels"],
                "image_id": anot["image_id"],
            }
        )

    metricas = evaluar_predicciones(
        predicciones,
        objetivos,
        incluir_pr=True,
        incluir_errores=True,
    )
    metricas["tiempos"] = {
        "ms_por_imagen": round((tiempo_s / max(imagenes_medidas, 1)) * 1000.0, 2),
        "fps": round(imagenes_medidas / max(tiempo_s, 1e-9), 2),
        "imagenes_medidas": imagenes_medidas,
    }
    metricas["modos_tta"] = list(modos)
    metricas["iou_wbf"] = iou_wbf
    return metricas, predicciones, objetivos


def ejecutar_ensemble(
    rutas_preds: list[Path],
    pesos: list[float] | None = None,
    iou_wbf: float = 0.55,
    skip_box_thr: float = 0.05,
    particion: str = "test",
) -> tuple[dict, list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Combina archivos JSON de predicciones de varios modelos mediante WBF y evalúa."""
    listas = [cargar_predicciones(ruta) for ruta in rutas_preds]
    preds_ens = combinar_predicciones_ensemble(
        listas,
        pesos=pesos,
        iou_wbf=iou_wbf,
        skip_box_thr=skip_box_thr,
    )
    ds = CorrosionDataset(particion)
    objetivos = [
        {
            "boxes": ds[i][1]["boxes"].float(),
            "labels": ds[i][1]["labels"],
            "image_id": ds[i][1]["image_id"],
        }
        for i in range(len(ds))
    ]
    metricas = evaluar_predicciones(
        preds_ens,
        objetivos,
        incluir_pr=True,
        incluir_errores=True,
    )
    metricas["modelos_ensemble"] = [str(r) for r in rutas_preds]
    metricas["pesos"] = pesos or [1.0] * len(rutas_preds)
    metricas["iou_wbf"] = iou_wbf
    return metricas, preds_ens, objetivos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa ajuste NMS/confianza, TTA y Ensemble WBF."
    )
    sub = parser.add_subparsers(dest="modo", required=True)
    dir_resultados = Path("results/slippage_validation_split")

    p_nms = sub.add_parser("nms", help="Barrido de NMS y umbral de confianza.")
    p_nms.add_argument("--checkpoint", type=Path, required=True)
    p_nms.add_argument("--salida", type=Path, default=dir_resultados / "exp06_postprocesado_nms")

    p_tta = sub.add_parser("tta", help="Evaluación con Test-Time Augmentation (WBF).")
    p_tta.add_argument("--checkpoint", type=Path, required=True)
    p_tta.add_argument(
        "--vistas",
        nargs="+",
        default=["original", "hflip", "vflip"],
        choices=["original", "hflip", "vflip", "hvflip"],
    )
    p_tta.add_argument("--iou-wbf", type=float, default=0.55)
    p_tta.add_argument("--salida", type=Path, default=dir_resultados / "exp07_tta_wbf")

    p_ens = sub.add_parser("ensemble", help="Ensemble WBF a partir de predicciones JSON.")
    p_ens.add_argument("--preds", type=Path, nargs="+", required=True)
    p_ens.add_argument("--pesos", type=float, nargs="+", default=None)
    p_ens.add_argument("--iou-wbf", type=float, default=0.55)
    p_ens.add_argument("--salida", type=Path, default=dir_resultados / "exp08_ensemble_wbf")

    args = parser.parse_args()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.modo == "nms":
        modelo, _ = cargar_modelo_desde_checkpoint(args.checkpoint, dispositivo)
        res = ejecutar_barrido_nms(
            modelo=modelo,
            dispositivo=dispositivo,
            umbrales_nms=[0.30, 0.40, 0.50, 0.60],
            umbrales_score=[0.01, 0.05, 0.15, 0.25],
        )
        preds_test = res.pop("_preds_test")
        objs_test = res.pop("_objs_test")
        args.salida.mkdir(parents=True, exist_ok=True)
        guardar_predicciones(args.salida / "predicciones_test.json", preds_test, objs_test)
        with open(args.salida / "metricas_test.json", "w", encoding="utf-8") as archivo:
            json.dump(res, archivo, indent=2, ensure_ascii=False)
        print(f"\nMejor config en val: {res['mejor_config']} (mAP={res['mejor_val_map']:.4f})")
        print("\n=== Resultados en test (NMS/score ajustado) ===")
        print(formatear_metricas(res["test"]))

    elif args.modo == "tta":
        modelo, _ = cargar_modelo_desde_checkpoint(args.checkpoint, dispositivo)
        metricas, preds, objs = ejecutar_tta(
            modelo=modelo,
            dispositivo=dispositivo,
            modos=tuple(args.vistas),
            iou_wbf=args.iou_wbf,
        )
        args.salida.mkdir(parents=True, exist_ok=True)
        guardar_predicciones(args.salida / "predicciones_test.json", preds, objs)
        with open(args.salida / "metricas_test.json", "w", encoding="utf-8") as archivo:
            json.dump({"test": metricas}, archivo, indent=2, ensure_ascii=False)
        print(f"\n=== Resultados en test (TTA: {'+'.join(args.vistas)}) ===")
        print(formatear_metricas(metricas))

    elif args.modo == "ensemble":
        metricas, preds, objs = ejecutar_ensemble(
            rutas_preds=args.preds,
            pesos=args.pesos,
            iou_wbf=args.iou_wbf,
        )
        args.salida.mkdir(parents=True, exist_ok=True)
        guardar_predicciones(args.salida / "predicciones_test.json", preds, objs)
        with open(args.salida / "metricas_test.json", "w", encoding="utf-8") as archivo:
            json.dump({"test": metricas}, archivo, indent=2, ensure_ascii=False)
        print("\n=== Resultados en test (Ensemble WBF) ===")
        print(formatear_metricas(metricas))


if __name__ == "__main__":
    main()
