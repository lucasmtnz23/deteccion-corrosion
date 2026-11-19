"""Entrenamiento (fine-tuning) de detectores de torchvision sobre Corrosion-bi3q3."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.aumentos import construir_transformaciones
from src.dataset import CorrosionDataset, collate_fn
from src.evaluar import evaluar_modelo, formatear_metricas, guardar_predicciones
from src.modelos import contar_parametros, crear_modelo
from src.modelos.faster_rcnn import (
    establecer_capas_entrenables_backbone,
    grupos_parametros_diferenciados,
)


def fijar_semilla(semilla: int) -> None:
    """Fija semillas para reproducibilidad."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def crear_optimizador(modelo: torch.nn.Module, cfg_ent: dict) -> torch.optim.Optimizer:
    """Construye el optimizador con soporte para LR diferenciada en el backbone."""
    nombre = str(cfg_ent.get("optimizador", "adamw")).strip().lower()
    lr = float(cfg_ent.get("lr", 1e-4))
    weight_decay = float(cfg_ent.get("weight_decay", 1e-4))
    factor_lr_backbone = float(cfg_ent.get("factor_lr_backbone", 1.0))

    if hasattr(modelo, "backbone") and hasattr(modelo.backbone, "body"):
        grupos = grupos_parametros_diferenciados(
            modelo,
            lr=lr,
            factor_lr_backbone=factor_lr_backbone,
            weight_decay=weight_decay,
        )
    else:
        grupos = [
            {
                "params": [p for p in modelo.parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

    if nombre == "adamw":
        return torch.optim.AdamW(grupos, lr=lr, weight_decay=weight_decay)
    if nombre == "sgd":
        momentum = float(cfg_ent.get("momentum", 0.9))
        return torch.optim.SGD(
            grupos,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Optimizador no soportado: {nombre!r}. Usa 'adamw' o 'sgd'.")


def crear_scheduler(
    optimizador: torch.optim.Optimizer,
    epocas: int,
    cfg_ent: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construye la política de tasa de aprendizaje (cosine o multistep con warmup)."""
    tipo = str(cfg_ent.get("scheduler", "cosine")).strip().lower()
    warmup_epocas = int(cfg_ent.get("warmup_epocas", 1))
    warmup_epocas = max(0, min(warmup_epocas, max(0, epocas - 1)))

    if tipo == "cosine":
        def factor_coseno(epoca: int) -> float:
            if warmup_epocas > 0 and epoca < warmup_epocas:
                return float(epoca + 1) / float(warmup_epocas + 1)
            progreso = (epoca - warmup_epocas) / max(1, epocas - warmup_epocas)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progreso))

        return torch.optim.lr_scheduler.LambdaLR(optimizador, lr_lambda=factor_coseno)

    if tipo == "multistep":
        milestones = [int(m) for m in cfg_ent.get("milestones", [int(epocas * 0.65), int(epocas * 0.85)])]
        gamma = float(cfg_ent.get("gamma", 0.1))

        def factor_multistep(epoca: int) -> float:
            if warmup_epocas > 0 and epoca < warmup_epocas:
                return float(epoca + 1) / float(warmup_epocas + 1)
            reducciones = sum(1 for m in milestones if epoca >= m)
            return gamma ** reducciones

        return torch.optim.lr_scheduler.LambdaLR(optimizador, lr_lambda=factor_multistep)

    raise ValueError(f"Scheduler no soportado: {tipo!r}. Usa 'cosine' o 'multistep'.")


def entrenar_epoca(
    modelo: torch.nn.Module,
    cargador: DataLoader,
    optimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
    scaler: torch.amp.GradScaler,
    usar_amp: bool,
    pasos_acumulacion: int = 1,
    clip_grad_norm: float | None = 10.0,
    epoca: int = 1,
    epocas_totales: int = 1,
) -> dict[str, float]:
    """Ejecuta una época de entrenamiento y devuelve el promedio de pérdidas."""
    modelo.train()
    amp_activo = usar_amp and dispositivo.type == "cuda"
    pasos_acumulacion = max(1, pasos_acumulacion)

    acumulado_perdidas: dict[str, float] = {}
    lotes_validos = 0
    optimizador.zero_grad(set_to_none=True)

    barra = tqdm(cargador, desc=f"Época {epoca}/{epocas_totales}", leave=False)
    for paso, (imagenes, anotaciones) in enumerate(barra, start=1):
        imagenes_gpu = [img.to(dispositivo) for img in imagenes]
        anotaciones_gpu = [
            {k: v.to(dispositivo) for k, v in anot.items()} for anot in anotaciones
        ]

        with torch.amp.autocast(device_type=dispositivo.type, enabled=amp_activo):
            dic_perdidas = modelo(imagenes_gpu, anotaciones_gpu)
            perdida_total = sum(dic_perdidas.values())

        valor_total = float(perdida_total.detach().item())
        if not math.isfinite(valor_total):
            print(f"[AVISO] Pérdida no finita ({valor_total}) en paso {paso}; lote omitido.")
            optimizador.zero_grad(set_to_none=True)
            continue

        perdida_escalada = perdida_total / pasos_acumulacion
        scaler.scale(perdida_escalada).backward()

        es_ultimo = paso == len(cargador)
        if paso % pasos_acumulacion == 0 or es_ultimo:
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in modelo.parameters() if p.requires_grad],
                    max_norm=clip_grad_norm,
                )
            scaler.step(optimizador)
            scaler.update()
            optimizador.zero_grad(set_to_none=True)

        lotes_validos += 1
        acumulado_perdidas["total"] = acumulado_perdidas.get("total", 0.0) + valor_total
        for clave, tensor_perdida in dic_perdidas.items():
            acumulado_perdidas[clave] = acumulado_perdidas.get(clave, 0.0) + float(
                tensor_perdida.detach().item()
            )
        barra.set_postfix(loss=f"{valor_total:.4f}")

    if lotes_validos == 0:
        raise RuntimeError("No se procesó ningún lote válido en la época.")

    return {k: v / lotes_validos for k, v in acumulado_perdidas.items()}


def entrenar(config: dict, evaluar_test: bool = True) -> Path:
    """Ejecuta el flujo completo de fine-tuning, validación y evaluación en test."""
    cfg_modelo = dict(config["modelo"])
    cfg_datos = config.get("datos", {})
    cfg_ent = config.get("entrenamiento", {})

    semilla = int(cfg_ent.get("semilla", 42))
    fijar_semilla(semilla)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type == "cuda":
        print(f"Entrenando en GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Entrenando en CPU (no se detectó GPU CUDA).")

    transform_train = construir_transformaciones(cfg_datos)
    ds_train = CorrosionDataset("train", transformacion=transform_train)
    ds_val = CorrosionDataset("validation")

    batch_size = int(cfg_datos.get("batch_size", 2))
    num_workers = int(cfg_datos.get("num_workers", 0))

    cargador_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(dispositivo.type == "cuda"),
    )
    cargador_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(dispositivo.type == "cuda"),
    )

    nombre_modelo = cfg_modelo.pop("nombre")
    print(f"Cargando modelo preentrenado: {nombre_modelo}...")
    modelo = crear_modelo(nombre_modelo, **cfg_modelo)
    modelo.to(dispositivo)

    totales, entrenables = contar_parametros(modelo)
    print(f"Parámetros totales: {totales:,} | Entrenables iniciales: {entrenables:,}")

    epocas = int(cfg_ent.get("epocas", 12))
    pasos_acumulacion = int(cfg_ent.get("pasos_acumulacion", 4))
    clip_grad_norm = cfg_ent.get("clip_grad_norm", 10.0)
    usar_amp = bool(cfg_ent.get("usar_amp", True))
    metrica_seleccion = str(cfg_ent.get("metrica_seleccion", "map"))

    # Descongelado progresivo opcional.
    descongelar_en_epoca = cfg_ent.get("descongelar_en_epoca", None)
    capas_tras_descongelar = int(cfg_ent.get("capas_tras_descongelar", 5))

    optimizador = crear_optimizador(modelo, cfg_ent)
    scheduler = crear_scheduler(optimizador, epocas=epocas, cfg_ent=cfg_ent)
    scaler = torch.amp.GradScaler("cuda", enabled=(usar_amp and dispositivo.type == "cuda"))

    dir_salida = Path(cfg_ent.get("dir_salida", f"results/{nombre_modelo}"))
    dir_salida.mkdir(parents=True, exist_ok=True)
    with open(dir_salida / "config_usada.yaml", "w", encoding="utf-8") as archivo:
        yaml.safe_dump(config, archivo, sort_keys=False, allow_unicode=True)

    ruta_mejor = dir_salida / "mejor_modelo.pth"
    ruta_ultimo = dir_salida / "ultimo_modelo.pth"

    mejor_valor = -1.0
    mejor_epoca = 0
    historial: list[dict] = []

    inicio_total = time.perf_counter()
    for epoca in range(1, epocas + 1):
        if descongelar_en_epoca is not None and epoca == int(descongelar_en_epoca):
            establecer_capas_entrenables_backbone(modelo, capas_tras_descongelar)
            totales, entrenables = contar_parametros(modelo)
            print(
                f"  -> Descongelado progresivo en época {epoca}: "
                f"{capas_tras_descongelar} bloques del backbone activos "
                f"({entrenables:,}/{totales:,} parámetros entrenables)."
            )
            optimizador = crear_optimizador(modelo, cfg_ent)
            epocas_restantes = epocas - epoca + 1
            cfg_rest = dict(cfg_ent, warmup_epocas=1)
            scheduler = crear_scheduler(optimizador, epocas=epocas_restantes, cfg_ent=cfg_rest)

        inicio_epoca = time.perf_counter()
        lr_actual = optimizador.param_groups[-1]["lr"]

        perdidas = entrenar_epoca(
            modelo=modelo,
            cargador=cargador_train,
            optimizador=optimizador,
            dispositivo=dispositivo,
            scaler=scaler,
            usar_amp=usar_amp,
            pasos_acumulacion=pasos_acumulacion,
            clip_grad_norm=clip_grad_norm,
            epoca=epoca,
            epocas_totales=epocas,
        )
        scheduler.step()

        metricas_val = evaluar_modelo(
            modelo=modelo,
            cargador=cargador_val,
            dispositivo=dispositivo,
            usar_amp=usar_amp,
            mostrar_progreso=False,
        )
        duracion = time.perf_counter() - inicio_epoca

        valor_actual = float(metricas_val[metrica_seleccion])
        registro = {
            "epoca": epoca,
            "lr": lr_actual,
            "duracion_s": round(duracion, 2),
            "perdidas_train": perdidas,
            "metricas_val": metricas_val,
        }
        historial.append(registro)

        print(
            f"Época {epoca:02d}/{epocas:02d} ({duracion:5.1f}s) | "
            f"lr={lr_actual:.2e} | "
            f"loss={perdidas['total']:.4f} | "
            f"val mAP={metricas_val['map']:.4f} | "
            f"mAP50={metricas_val['map_50']:.4f} | "
            f"mAP_s={metricas_val['map_small']:.4f}"
        )

        estado_ckpt = {
            "epoca": epoca,
            "config": config,
            "modelo_state_dict": modelo.state_dict(),
            "optimizador_state_dict": optimizador.state_dict(),
            "metricas_val": metricas_val,
        }
        torch.save(estado_ckpt, ruta_ultimo)

        if valor_actual > mejor_valor:
            mejor_valor = valor_actual
            mejor_epoca = epoca
            torch.save(estado_ckpt, ruta_mejor)
            print(f"  -> Nuevo mejor modelo guardado ({metrica_seleccion}={mejor_valor:.4f})")

        with open(dir_salida / "historial.json", "w", encoding="utf-8") as archivo:
            json.dump(historial, archivo, indent=2, ensure_ascii=False)

    tiempo_total = time.perf_counter() - inicio_total
    print(
        f"\nEntrenamiento completado en {tiempo_total / 60:.1f} min. "
        f"Mejor época: {mejor_epoca} ({metrica_seleccion}={mejor_valor:.4f})."
    )

    if evaluar_test and ruta_mejor.exists():
        print("\nEvaluando el mejor checkpoint sobre la partición de test (con PR y latencia)...")
        ckpt_mejor = torch.load(ruta_mejor, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(ckpt_mejor["modelo_state_dict"])

        ds_test = CorrosionDataset("test")
        cargador_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(dispositivo.type == "cuda"),
        )
        metricas_test, preds_test, objs_test = evaluar_modelo(
            modelo=modelo,
            cargador=cargador_test,
            dispositivo=dispositivo,
            usar_amp=usar_amp,
            detallado=True,
            devolver_predicciones=True,
        )
        print("\n=== Resultados en test (mejor modelo) ===")
        print(formatear_metricas(metricas_test))

        guardar_predicciones(dir_salida / "predicciones_test.json", preds_test, objs_test)
        with open(dir_salida / "metricas_test.json", "w", encoding="utf-8") as archivo:
            json.dump(
                {
                    "mejor_epoca": mejor_epoca,
                    "metrica_seleccion": metrica_seleccion,
                    "mejor_val": mejor_valor,
                    "tiempo_entrenamiento_min": round(tiempo_total / 60.0, 2),
                    "test": metricas_test,
                },
                archivo,
                indent=2,
                ensure_ascii=False,
            )

    return ruta_mejor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de modelos de detección sobre Corrosion-bi3q3."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/faster_rcnn.yaml"),
        help="Ruta al archivo de configuración YAML.",
    )
    parser.add_argument("--epocas", type=int, default=None, help="Sobrescribe el número de épocas.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sobrescribe el batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Sobrescribe el learning rate.")
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Sobrescribe el directorio de salida de checkpoints y métricas.",
    )
    parser.add_argument(
        "--sin-test",
        action="store_true",
        help="Omite la evaluación final sobre la partición de test.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)

    if args.epocas is not None:
        config.setdefault("entrenamiento", {})["epocas"] = args.epocas
    if args.batch_size is not None:
        config.setdefault("datos", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("entrenamiento", {})["lr"] = args.lr
    if args.salida is not None:
        config.setdefault("entrenamiento", {})["dir_salida"] = str(args.salida)

    entrenar(config, evaluar_test=not args.sin_test)


if __name__ == "__main__":
    main()
