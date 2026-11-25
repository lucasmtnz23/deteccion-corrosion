"""Ejecuta en validación las variantes que faltan para comparar ambos detectores."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from pickle import UnpicklingError

import torch
import yaml

from scripts.evaluar_postprocesado import crear_parser as crear_parser_postprocesado
from src.entrenar import entrenar
from src.procedencia import actualizar_validacion, identificar_archivo, procedencia_validacion


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


def comprobar_checkpoint(ruta: Path, config: dict) -> int:
    """Contrasta la configuración del propio checkpoint sin cargar sus tensores en memoria."""
    try:
        estado = torch.load(ruta, map_location="cpu", mmap=True, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, UnpicklingError) as error:
        raise RuntimeError(f"{ruta}: no se pudo verificar el checkpoint.") from error
    if not isinstance(estado, dict) or estado.get("config") != config:
        raise ValueError(
            f"{ruta}: configuración incompatible o ausente. Conserva el checkpoint y usa otra salida."
        )
    if type(estado.get("epoca")) is not int:
        raise ValueError(f"{ruta}: falta una época válida en el checkpoint.")
    return estado["epoca"]


def preparar_entrenamiento(config: dict, *, solo_evaluacion: bool = False) -> Path:
    parametros = config["entrenamiento"]
    directorio = Path(parametros["dir_salida"])
    mejor = directorio / "mejor_modelo.pth"
    ultimo = directorio / "ultimo_modelo.pth"
    historial = directorio / "historial.json"
    ruta_config = directorio / "config_usada.yaml"
    hay_estado = any(ruta.exists() for ruta in (mejor, ultimo, historial))
    if hay_estado or ruta_config.exists():
        if not ruta_config.is_file() or yaml.safe_load(ruta_config.read_text(encoding="utf-8")) != config:
            raise ValueError(
                f"{ruta_config}: configuración de entrenamiento incompatible o ausente. "
                "Conserva los archivos del entrenamiento y usa otro directorio de salida."
            )

    epocas = json.loads(historial.read_text(encoding="utf-8")) if historial.exists() else []
    ultima_epoca = epocas[-1]["epoca"] if epocas else 0
    if mejor.is_file():
        comprobar_checkpoint(mejor, config)
        if ultima_epoca >= parametros["epocas"]:
            return directorio

    if solo_evaluacion:
        raise RuntimeError(f"{directorio}: se requiere un entrenamiento completo para --solo-evaluacion.")

    if hay_estado and not (
        mejor.is_file() and ultimo.is_file() and ultima_epoca > 0
        and comprobar_checkpoint(ultimo, config) == ultima_epoca
    ):
        raise RuntimeError(
            f"{directorio}: los checkpoints y el historial no permiten una reanudación segura. "
            "Conserva estos archivos y usa otro directorio de salida."
        )
    entrenar(config, evaluar_test=False, reanudar=hay_estado)
    return directorio


def entrenar_y_evaluar(
    config: dict, *, forzar_evaluacion: bool = False, solo_evaluacion: bool = False,
) -> Path:
    directorio = preparar_entrenamiento(config, solo_evaluacion=solo_evaluacion)
    checkpoint = directorio / "mejor_modelo.pth"
    batch_size, num_workers = 4, 0

    def obtener_procedencia() -> dict:
        return procedencia_validacion(
            "evaluacion", checkpoint=identificar_archivo(checkpoint),
            parametros={"batch_size": batch_size, "num_workers": num_workers},
        )

    def generar(salida: Path) -> None:
        ejecutar(
            "src.evaluar", "--checkpoint", checkpoint, "--particion", "validation",
            "--batch-size", str(batch_size), "--num-workers", str(num_workers),
            "--guardar", salida / "metricas_validation.json",
            "--guardar-preds", salida / "predicciones_validation.json",
        )

    actualizar_validacion(directorio, obtener_procedencia, generar, forzar_evaluacion=forzar_evaluacion)
    return directorio


def postprocesar(modo: str, nombre: str, *args: str | Path, forzar_evaluacion: bool = False) -> None:
    salida = DIR_RESULTADOS / nombre
    argumentos = [modo, *map(str, args)]
    opciones = vars(crear_parser_postprocesado().parse_args(argumentos))
    if opciones["particion"] != "validation" or opciones.get("evaluar_test"):
        raise ValueError("Los experimentos cruzados solo generan resultados en validación.")
    parametros = {
        clave: valor for clave, valor in opciones.items()
        if clave not in {"modo", "salida", "checkpoint", "preds", "particion"}
    }

    def obtener_procedencia() -> dict:
        if modo == "ensemble":
            rutas = opciones["preds"]
            return procedencia_validacion(
                modo, predicciones=[identificar_archivo(ruta) for ruta in rutas],
                parametros={**parametros, "pesos": opciones["pesos"] or [1.0] * len(rutas)},
            )
        return procedencia_validacion(
            modo, checkpoint=identificar_archivo(opciones["checkpoint"]), parametros=parametros,
        )

    def generar(temporal: Path) -> None:
        ejecutar("scripts.evaluar_postprocesado", *argumentos, "--salida", temporal)

    actualizar_validacion(salida, obtener_procedencia, generar, forzar_evaluacion=forzar_evaluacion)


def fusionar_retinanet(directorios: list[Path], *, forzar_evaluacion: bool = False) -> None:
    rutas = [directorio / "predicciones_validation.json" for directorio in directorios]
    postprocesar(
        "ensemble", "retinanet_ensemble_wbf", "--particion", "validation",
        "--preds", *rutas, "--pesos", "1", "1", "1", "--iou-wbf", "0.55",
        forzar_evaluacion=forzar_evaluacion,
    )


def ejecutar_experimentos(
    modelo: str, *, forzar_evaluacion: bool = False, solo_evaluacion: bool = False,
) -> None:
    evaluar = partial(
        entrenar_y_evaluar, forzar_evaluacion=forzar_evaluacion, solo_evaluacion=solo_evaluacion,
    )
    procesar = partial(postprocesar, forzar_evaluacion=forzar_evaluacion)
    nuevos = {
        destino: evaluar(config_cruzada(origen, modelo, destino))
        for origen, destino in EXPERIMENTOS_CRUZADOS[modelo]
    }
    if modelo == "retinanet":
        e2 = evaluar(cargar_config("retinanet_exp02_resolucion_20ep"))
        checkpoint = e2 / "mejor_modelo.pth"
        procesar("nms", "retinanet_nms_e2", "--checkpoint", checkpoint)
        procesar(
            "tta", "retinanet_exp02_resolucion_20ep_tta", "--checkpoint", checkpoint,
            "--particion", "validation", "--vistas", "original", "hflip",
        )
        procesar(
            "tta", "retinanet_tta_3_vistas", "--checkpoint", checkpoint,
            "--particion", "validation", "--vistas", "original", "hflip", "vflip",
        )
        aumentos = evaluar(cargar_config("retinanet_exp01_aumentos"))
        fusionar_retinanet([
            aumentos, nuevos["retinanet_anchors_resolucion"], nuevos["retinanet_perdidas_ciou_ponderada"],
        ], forzar_evaluacion=forzar_evaluacion)
    else:
        aumentos = evaluar(cargar_config("01_data_augmentation"))
        procesar(
            "tta", "faster_rcnn_tta_2_vistas", "--checkpoint", aumentos / "mejor_modelo.pth",
            "--particion", "validation", "--vistas", "original", "hflip",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, choices=(*EXPERIMENTOS_CRUZADOS, "ambos"))
    parser.add_argument(
        "--forzar-evaluacion", action="store_true",
        help="Regenera las evaluaciones y el postprocesado en validación.",
    )
    parser.add_argument(
        "--solo-evaluacion", action="store_true",
        help="Exige checkpoints completos y compatibles e impide entrenar o reanudar.",
    )
    args = parser.parse_args()
    for nombre in EXPERIMENTOS_CRUZADOS if args.modelo == "ambos" else (args.modelo,):
        ejecutar_experimentos(
            nombre, forzar_evaluacion=args.forzar_evaluacion, solo_evaluacion=args.solo_evaluacion,
        )


if __name__ == "__main__":
    main()
