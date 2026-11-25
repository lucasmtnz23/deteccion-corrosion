"""Ejecuta en validación las variantes que faltan para comparar ambos detectores."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.entrenar import entrenar


DIR_CONFIGS = Path("configs/experimentos")
DIR_RESULTADOS = Path("results/slippage_validation_split")

EXPERIMENTOS_CRUZADOS = {
    "retinanet": (
        ("02_transfer_learning", "retinanet_transfer_learning"),
        ("03_sgd_optimizacion", "retinanet_sgd_multistep"),
        ("04_anchors_resolucion", "retinanet_anchors_resolucion"),
        ("05_perdidas_ciou_ponderada", "retinanet_perdidas_ciou_ponderada"),
    ),
    "faster_rcnn": (
        ("retinanet_exp02_resolucion", "faster_rcnn_resolucion_12ep"),
        ("retinanet_exp02_resolucion_20ep", "faster_rcnn_resolucion_20ep"),
        ("retinanet_exp03_sgd", "faster_rcnn_sgd_cosine"),
    ),
}


def cargar_config(nombre: str) -> dict:
    with (DIR_CONFIGS / f"{nombre}.yaml").open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def config_cruzada(origen: str, modelo: str, destino: str) -> dict:
    config = cargar_config(origen)
    config["modelo"]["nombre"] = modelo
    config["entrenamiento"]["dir_salida"] = str(DIR_RESULTADOS / destino)
    perdida = config["modelo"].get("perdida")
    if perdida and modelo == "retinanet":
        # RetinaNet usa clasificación focal sigmoide de forma nativa.
        perdida["clasificacion"] = "focal"
    return config


def ejecutar(modulo: str, *args: str | Path) -> None:
    subprocess.run([sys.executable, "-m", modulo, *map(str, args)], check=True)


def hay_validacion(directorio: Path) -> bool:
    return all(
        (directorio / nombre).is_file()
        for nombre in ("metricas_validation.json", "predicciones_validation.json")
    )


def entrenar_y_evaluar(config: dict) -> Path:
    parametros = config["entrenamiento"]
    directorio = Path(parametros["dir_salida"])
    checkpoint = directorio / "mejor_modelo.pth"
    historial = directorio / "historial.json"
    epocas = json.loads(historial.read_text(encoding="utf-8")) if historial.is_file() else []
    completo = checkpoint.is_file() and epocas and epocas[-1]["epoca"] >= parametros["epocas"]
    if not completo:
        reanudar = checkpoint.is_file() and (directorio / "ultimo_modelo.pth").is_file()
        entrenar(config, evaluar_test=False, reanudar=reanudar)

    if not completo or not hay_validacion(directorio):
        ejecutar(
            "src.evaluar", "--checkpoint", checkpoint, "--particion", "validation",
            "--batch-size", "4", "--guardar", directorio / "metricas_validation.json",
            "--guardar-preds", directorio / "predicciones_validation.json",
        )
    return directorio


def postprocesar(modo: str, nombre: str, *args: str | Path) -> None:
    salida = DIR_RESULTADOS / nombre
    if not hay_validacion(salida):
        ejecutar("scripts.evaluar_postprocesado", modo, *args, "--salida", salida)


def fusionar_retinanet(directorios: list[Path]) -> None:
    rutas = [directorio / "predicciones_validation.json" for directorio in directorios]
    ids = [
        [fila["image_id"] for fila in json.loads(ruta.read_text(encoding="utf-8"))]
        for ruta in rutas
    ]
    if len(ids[0]) != len(set(ids[0])) or any(orden != ids[0] for orden in ids[1:]):
        raise ValueError("Los JSON del ensemble deben tener los mismos IDs únicos en el mismo orden.")
    postprocesar(
        "ensemble", "retinanet_ensemble_wbf", "--particion", "validation",
        "--preds", *rutas, "--pesos", "1", "1", "1", "--iou-wbf", "0.55",
    )


def ejecutar_experimentos(modelo: str) -> None:
    nuevos = {
        destino: entrenar_y_evaluar(config_cruzada(origen, modelo, destino))
        for origen, destino in EXPERIMENTOS_CRUZADOS[modelo]
    }
    if modelo == "retinanet":
        e2 = entrenar_y_evaluar(cargar_config("retinanet_exp02_resolucion_20ep"))
        checkpoint = e2 / "mejor_modelo.pth"
        postprocesar("nms", "retinanet_nms_e2", "--checkpoint", checkpoint)
        postprocesar(
            "tta", "retinanet_exp02_resolucion_20ep_tta", "--checkpoint", checkpoint,
            "--particion", "validation", "--vistas", "original", "hflip",
        )
        postprocesar(
            "tta", "retinanet_tta_3_vistas", "--checkpoint", checkpoint,
            "--particion", "validation", "--vistas", "original", "hflip", "vflip",
        )
        aumentos = entrenar_y_evaluar(cargar_config("retinanet_exp01_aumentos"))
        fusionar_retinanet([
            aumentos, nuevos["retinanet_anchors_resolucion"], nuevos["retinanet_perdidas_ciou_ponderada"],
        ])
    else:
        aumentos = entrenar_y_evaluar(cargar_config("01_data_augmentation"))
        postprocesar(
            "tta", "faster_rcnn_tta_2_vistas", "--checkpoint", aumentos / "mejor_modelo.pth",
            "--particion", "validation", "--vistas", "original", "hflip",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, choices=(*EXPERIMENTOS_CRUZADOS, "ambos"))
    modelo = parser.parse_args().modelo
    for nombre in EXPERIMENTOS_CRUZADOS if modelo == "ambos" else (modelo,):
        ejecutar_experimentos(nombre)


if __name__ == "__main__":
    main()
