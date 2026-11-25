"""Huellas y conservación de los resultados de validación de los experimentos cruzados."""

from __future__ import annotations

import json
import shutil
import tempfile
from hashlib import file_digest
from pathlib import Path
from typing import Callable

from src import dataset


ARCHIVOS_VALIDACION = ("metricas_validation.json", "predicciones_validation.json")
ARCHIVO_PROCEDENCIA = "procedencia_validation.json"


def huella_archivo(ruta: Path) -> str:
    with ruta.open("rb") as archivo:
        return file_digest(archivo, "sha256").hexdigest()


def identificar_archivo(ruta: Path) -> dict:
    return {"ruta": str(ruta), "sha256": huella_archivo(ruta)}


def procedencia_validacion(modo: str, **entradas) -> dict:
    """Identifica el manifiesto y las particiones de origen que forman validación."""
    raiz = dataset.RUTA_DATOS
    archivos = [raiz / "dataset_dict.json"]
    for particion in ("train", "validation"):
        carpeta = raiz / particion
        estado = carpeta / "state.json"
        fragmentos = json.loads(estado.read_text(encoding="utf-8"))["_data_files"]
        archivos.extend((estado, carpeta / "dataset_info.json"))
        archivos.extend(carpeta / fragmento["filename"] for fragmento in fragmentos)
    return {
        "version": 1,
        "modo": modo,
        "particion": "validation",
        "dataset": {
            "manifiesto": identificar_archivo(dataset.RUTA_MANIFIESTO),
            "archivos": {ruta.relative_to(raiz).as_posix(): huella_archivo(ruta) for ruta in archivos},
            "clases": {str(k): v for k, v in dataset.NOMBRES_CLASES.items()},
        },
        **entradas,
    }


def _huellas_resultados(directorio: Path) -> dict:
    return {nombre: huella_archivo(directorio / nombre) for nombre in ARCHIVOS_VALIDACION}


def _motivo_regeneracion(directorio: Path, esperada: dict) -> str | None:
    ruta = directorio / ARCHIVO_PROCEDENCIA
    try:
        guardada = json.loads(ruta.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        if any((directorio / nombre).exists() for nombre in (*ARCHIVOS_VALIDACION, ARCHIVO_PROCEDENCIA)):
            return "resultados previos sin procedencia verificable"
        return "sin resultados previos"
    if not isinstance(guardada, dict) or guardada.get("entradas") != esperada:
        return "procedencia distinta o incompleta"
    try:
        if guardada.get("resultados") == _huellas_resultados(directorio):
            return None
    except FileNotFoundError:
        pass
    return "métricas o predicciones ausentes o modificadas"


def actualizar_validacion(
    directorio: Path,
    obtener_procedencia: Callable[[], dict],
    generar: Callable[[Path], None],
    *,
    forzar_evaluacion: bool = False,
) -> None:
    esperada = obtener_procedencia()
    motivo = _motivo_regeneracion(directorio, esperada)
    if motivo is None and not forzar_evaluacion:
        print(f"{directorio}: reutilizando validación compatible.", flush=True)
        return

    print(f"{directorio}: regenerando validación ({motivo or 'evaluación forzada'}).", flush=True)
    directorio.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validacion-", dir=directorio) as temporal:
        salida = Path(temporal)
        generar(salida)
        if obtener_procedencia() != esperada:
            raise RuntimeError(f"{directorio}: cambiaron las entradas durante la evaluación.")
        registro = {"entradas": esperada, "resultados": _huellas_resultados(salida)}
        (salida / ARCHIVO_PROCEDENCIA).write_text(
            json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        nombres = (*ARCHIVOS_VALIDACION, ARCHIVO_PROCEDENCIA)
        anteriores = [directorio / nombre for nombre in nombres if (directorio / nombre).exists()]
        if anteriores:
            historico = directorio / "validaciones_anteriores"
            historico.mkdir(exist_ok=True)
            respaldo = Path(tempfile.mkdtemp(prefix="validacion-", dir=historico))
            print(
                f"{directorio}: guardando los resultados previos en {respaldo} antes de reemplazarlos.",
                flush=True,
            )
            for ruta in anteriores:
                shutil.copy2(ruta, respaldo / ruta.name)
        # La procedencia se publica al final y sus huellas detectan una publicación incompleta.
        for nombre in nombres:
            (salida / nombre).replace(directorio / nombre)
