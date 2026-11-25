"""Pruebas de integridad de los ensembles y de reutilización en validación."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from scripts import comparar_experimentos as comparar
from scripts import evaluar_postprocesado as postprocesado
from src.evaluar import cargar_predicciones, cargar_predicciones_alineadas


ARCHIVOS_VALIDACION = ("metricas_validation.json", "predicciones_validation.json")


def escribir_json(ruta, valor):
    ruta.write_text(json.dumps(valor), encoding="utf-8")


class PruebasAlineacion(unittest.TestCase):
    def setUp(self):
        self.raiz = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ruta = self.raiz / "predicciones.json"
        self.filas = [
            {"image_id": 9, "boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": [1]},
            {"image_id": 4, "boxes": [[20, 0, 25, 5]], "scores": [0.8], "labels": [1]},
        ]
        self.objetivos = [
            {
                "image_id": torch.tensor(fila["image_id"]),
                "boxes": torch.tensor(fila["boxes"], dtype=torch.float32),
                "labels": torch.tensor(fila["labels"]),
            }
            for fila in self.filas
        ]

    def test_carga_ordenada_y_compatibilidad_del_cargador_original(self):
        escribir_json(self.ruta, self.filas)
        alineadas = cargar_predicciones_alineadas(self.ruta, self.objetivos)
        for objetivo, prediccion in zip(self.objetivos, alineadas, strict=True):
            torch.testing.assert_close(prediccion["boxes"], objetivo["boxes"])

        escribir_json(self.ruta, self.filas[::-1])
        torch.testing.assert_close(cargar_predicciones(self.ruta)[0]["boxes"], self.objetivos[1]["boxes"])

    def test_ensemble_invariante_al_orden_de_los_json(self):
        otra_ruta = self.raiz / "otras.json"
        escribir_json(self.ruta, self.filas)
        escribir_json(otra_ruta, self.filas)
        datos = [(None, objetivo) for objetivo in self.objetivos]
        with patch.object(postprocesado, "CorrosionDataset", return_value=datos), patch.object(
            postprocesado, "evaluar_predicciones", return_value={},
        ):
            _, ordenadas, _ = postprocesado.ejecutar_ensemble([self.ruta, otra_ruta], particion="validation")
            escribir_json(otra_ruta, self.filas[::-1])
            _, desordenadas, _ = postprocesado.ejecutar_ensemble([self.ruta, otra_ruta], particion="validation")

        for esperado, original, reordenado in zip(self.objetivos, ordenadas, desordenadas, strict=True):
            torch.testing.assert_close(original["boxes"], esperado["boxes"])
            torch.testing.assert_close(original, reordenado)

    def test_ids_duplicados_ausentes_e_inesperados(self):
        casos = (
            ([self.filas[0]] * 2, r"duplicados: \[9\], ausentes: \[4\]"),
            (self.filas[:1], r"ausentes: \[4\]"),
            ([self.filas[0], {**self.filas[1], "image_id": 17}], r"ausentes: \[4\], inesperados: \[17\]"),
            ([{"boxes": [], "scores": [], "labels": []}], r"image_id.*ausente.*registro 1"),
        )
        for filas, error in casos:
            with self.subTest(error=error):
                escribir_json(self.ruta, filas)
                with self.assertRaisesRegex(ValueError, rf"predicciones\.json: .*{error}"):
                    cargar_predicciones_alineadas(self.ruta, self.objetivos)


class PruebasReutilizacion(unittest.TestCase):
    def setUp(self):
        self.raiz = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.salida = self.raiz / "modelo"
        self.checkpoint = self.salida / "mejor_modelo.pth"
        self.config = self.crear_entrenamiento(self.salida)

        self.enterContext(patch.object(comparar, "DIR_RESULTADOS", self.raiz))
        self.entrenar = self.enterContext(patch.object(comparar, "entrenar"))
        self.ejecutar = self.enterContext(patch.object(comparar, "ejecutar", side_effect=self.generar))

    def crear_entrenamiento(self, salida):
        salida.mkdir()
        config = {"modelo": {"nombre": "retinanet"}, "entrenamiento": {"epocas": 2, "dir_salida": str(salida)}}
        (salida / "config_usada.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        escribir_json(salida / "historial.json", [{"epoca": 1}, {"epoca": 2}])
        self.guardar_checkpoint(salida / "mejor_modelo.pth", config)
        return config

    @staticmethod
    def guardar_checkpoint(ruta, config):
        torch.save({"epoca": 1, "config": config}, ruta)

    def generar(self, modulo, *args):
        if modulo == "src.evaluar":
            salida = args[args.index("--guardar") + 1].parent
        else:
            salida = args[args.index("--salida") + 1]
        salida.mkdir(parents=True, exist_ok=True)
        escribir_json(salida / ARCHIVOS_VALIDACION[0], {"ejecucion": self.ejecutar.call_count})
        escribir_json(salida / ARCHIVOS_VALIDACION[1], [])

    def test_reutiliza_ambos_json_y_regenera_si_falta_uno(self):
        for nombre in ARCHIVOS_VALIDACION:
            escribir_json(self.salida / nombre, [])
        comparar.entrenar_y_evaluar(self.config)
        self.ejecutar.assert_not_called()

        for nombre in ARCHIVOS_VALIDACION:
            with self.subTest(archivo=nombre):
                self.ejecutar.reset_mock()
                (self.salida / nombre).unlink()
                comparar.entrenar_y_evaluar(self.config)
                self.ejecutar.assert_called_once()
                self.assertTrue((self.salida / nombre).is_file())
        self.entrenar.assert_not_called()

    def test_postprocesado_regenera_si_falta_un_json(self):
        args = ("--checkpoint", self.checkpoint)
        comparar.postprocesar("nms", "nms", *args)
        comparar.postprocesar("nms", "nms", *args)
        self.ejecutar.assert_called_once()
        (self.raiz / "nms" / "predicciones_validation.json").unlink()
        comparar.postprocesar("nms", "nms", *args)
        self.assertEqual(self.ejecutar.call_count, 2)

    def test_entrenamiento_nuevo_regenera_el_resultado_existente(self):
        salida = self.raiz / "nuevo"
        salida.mkdir()
        config = copy.deepcopy(self.config)
        config["entrenamiento"]["dir_salida"] = str(salida)
        for nombre in ARCHIVOS_VALIDACION:
            escribir_json(salida / nombre, [])

        with self.assertRaisesRegex(RuntimeError, "entrenamiento completo"):
            comparar.entrenar_y_evaluar(config, solo_evaluacion=True)
        self.entrenar.assert_not_called()
        self.ejecutar.assert_not_called()

        comparar.entrenar_y_evaluar(config)
        self.entrenar.assert_called_once_with(config, evaluar_test=False, reanudar=False)
        self.ejecutar.assert_called_once()
        self.assertEqual(json.loads((salida / "metricas_validation.json").read_text()), {"ejecucion": 1})

    def test_configuracion_incompatible_en_yaml_mejor_checkpoint_o_reanudacion(self):
        incompatible = copy.deepcopy(self.config)
        incompatible["modelo"]["nombre"] = "faster_rcnn"
        with self.assertRaisesRegex(ValueError, "config_usada.yaml: configuración.*incompatible"):
            comparar.entrenar_y_evaluar(incompatible, forzar_evaluacion=True)
        self.guardar_checkpoint(self.checkpoint, incompatible)
        with self.assertRaisesRegex(ValueError, "mejor_modelo.pth: configuración incompatible"):
            comparar.entrenar_y_evaluar(self.config)

        self.guardar_checkpoint(self.checkpoint, self.config)
        escribir_json(self.salida / "historial.json", [{"epoca": 1}])
        ultimo = self.salida / "ultimo_modelo.pth"
        self.guardar_checkpoint(ultimo, incompatible)
        with self.assertRaisesRegex(ValueError, "ultimo_modelo.pth: configuración incompatible"):
            comparar.entrenar_y_evaluar(self.config)
        self.entrenar.assert_not_called()
        self.ejecutar.assert_not_called()

    def test_cli_forzar_evalua_y_postprocesa_sin_reentrenar(self):
        destinos = ("retinanet_anchors_resolucion", "retinanet_perdidas_ciou_ponderada")
        cruzados = {"retinanet": tuple((nombre, nombre) for nombre in destinos)}
        configs = {
            nombre: self.crear_entrenamiento(self.raiz / nombre)
            for nombre in (*destinos, "retinanet_exp02_resolucion_20ep")
        }
        configs["retinanet_exp01_aumentos"] = self.config
        args = ["comparar_experimentos", "--modelo", "retinanet", "--solo-evaluacion"]
        with patch.object(comparar, "EXPERIMENTOS_CRUZADOS", cruzados), patch.object(
            comparar, "cargar_config", side_effect=configs.__getitem__,
        ), patch.object(sys, "argv", args):
            comparar.main()
            comparar.main()
            self.assertEqual(self.ejecutar.call_count, 8)
            args.append("--forzar-evaluacion")
            comparar.main()
            self.assertEqual(self.ejecutar.call_count, 16)
            incompleto = self.raiz / destinos[0]
            escribir_json(incompleto / "historial.json", [{"epoca": 1}])
            self.guardar_checkpoint(incompleto / "ultimo_modelo.pth", configs[destinos[0]])
            with self.assertRaisesRegex(RuntimeError, "entrenamiento completo"):
                comparar.main()
            self.entrenar.assert_not_called()
            self.assertEqual(self.ejecutar.call_count, 16)

            args.remove("--solo-evaluacion")
            args.remove("--forzar-evaluacion")
            self.ejecutar.reset_mock()
            comparar.main()
            self.entrenar.assert_called_once_with(configs[destinos[0]], evaluar_test=False, reanudar=True)
            evaluacion, fusion = self.ejecutar.call_args_list
            self.assertEqual(evaluacion.args[0], "src.evaluar")
            self.assertEqual(fusion.args[:2], ("scripts.evaluar_postprocesado", "ensemble"))


if __name__ == "__main__":
    unittest.main()
