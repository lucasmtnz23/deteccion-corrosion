"""Pruebas de integridad de los ensembles y de reutilización en validación."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from scripts import comparar_experimentos as comparar
from scripts import evaluar_postprocesado as postprocesado
from src import dataset
from src.evaluar import cargar_predicciones, cargar_predicciones_alineadas
from src.procedencia import ARCHIVO_PROCEDENCIA, ARCHIVOS_VALIDACION


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

        datos = self.raiz / "datos"
        datos.mkdir()
        escribir_json(datos / "dataset_dict.json", {"splits": ["train", "validation"]})
        for particion in ("train", "validation"):
            carpeta = datos / particion
            carpeta.mkdir()
            escribir_json(carpeta / "state.json", {"_data_files": [{"filename": "datos.arrow"}]})
            escribir_json(carpeta / "dataset_info.json", {})
            (carpeta / "datos.arrow").write_bytes(particion.encode())
        self.manifiesto = self.raiz / "manifiesto.json"
        escribir_json(self.manifiesto, {"image_ids_train": [1]})

        self.enterContext(patch.object(dataset, "RUTA_DATOS", datos))
        self.enterContext(patch.object(dataset, "RUTA_MANIFIESTO", self.manifiesto))
        self.enterContext(patch.object(comparar, "DIR_RESULTADOS", self.raiz))
        self.entrenar = self.enterContext(patch.object(comparar, "entrenar"))
        self.addCleanup(self.entrenar.assert_not_called)
        self.ejecutar = self.enterContext(patch.object(comparar, "ejecutar", side_effect=self.generar))
        self.mensajes = self.enterContext(redirect_stdout(io.StringIO()))

    def crear_entrenamiento(self, salida):
        salida.mkdir()
        config = {"modelo": {"nombre": "retinanet"}, "entrenamiento": {"epocas": 2, "dir_salida": str(salida)}}
        (salida / "config_usada.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        escribir_json(salida / "historial.json", [{"epoca": 1}, {"epoca": 2}])
        self.guardar_checkpoint(salida / "mejor_modelo.pth", config)
        return config

    @staticmethod
    def guardar_checkpoint(ruta, config, revision=1):
        torch.save({"epoca": 1, "config": config, "modelo_state_dict": {"peso": torch.tensor(revision)}}, ruta)

    def generar(self, modulo, *args):
        if modulo == "src.evaluar":
            salida = args[args.index("--guardar") + 1].parent
        else:
            salida = args[args.index("--salida") + 1]
        escribir_json(salida / ARCHIVOS_VALIDACION[0], {"ejecucion": self.ejecutar.call_count})
        escribir_json(salida / ARCHIVOS_VALIDACION[1], [])

    def test_reutiliza_y_regenera_al_cambiar_checkpoint_dataset_o_salidas(self):
        comparar.entrenar_y_evaluar(self.config)
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 1)
        self.guardar_checkpoint(self.checkpoint, self.config, revision=2)
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 2)
        escribir_json(self.manifiesto, {"image_ids_train": [2]})
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 3)
        (dataset.RUTA_DATOS / "validation" / "datos.arrow").write_bytes(b"anotaciones distintas")
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 4)
        escribir_json(self.salida / ARCHIVOS_VALIDACION[0], {"editado": True})
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 5)

    def test_resultados_antiguos_se_conservan_y_solo_se_certifica_la_nueva_evaluacion(self):
        for nombre in ARCHIVOS_VALIDACION:
            escribir_json(self.salida / nombre, {"antiguo": True})
        comparar.entrenar_y_evaluar(self.config)
        self.assertEqual(self.ejecutar.call_count, 1)
        respaldo, = (self.salida / "validaciones_anteriores").iterdir()
        self.assertFalse((respaldo / ARCHIVO_PROCEDENCIA).exists())
        for nombre in ARCHIVOS_VALIDACION:
            self.assertEqual(json.loads((respaldo / nombre).read_text()), {"antiguo": True})
        self.assertTrue((self.salida / ARCHIVO_PROCEDENCIA).is_file())
        self.assertIn("sin procedencia verificable", self.mensajes.getvalue())

    def test_cambio_de_entradas_durante_generacion_conserva_resultados(self):
        comparar.entrenar_y_evaluar(self.config)
        anteriores = {
            nombre: (self.salida / nombre).read_bytes()
            for nombre in (*ARCHIVOS_VALIDACION, ARCHIVO_PROCEDENCIA)
        }

        def generar_con_cambio(modulo, *args):
            self.generar(modulo, *args)
            self.guardar_checkpoint(self.checkpoint, self.config, revision=2)

        self.ejecutar.side_effect = generar_con_cambio
        with self.assertRaisesRegex(RuntimeError, "cambiaron las entradas"):
            comparar.entrenar_y_evaluar(self.config, forzar_evaluacion=True)
        for nombre, contenido in anteriores.items():
            self.assertEqual((self.salida / nombre).read_bytes(), contenido)

    def test_postprocesado_por_checkpoint_y_parametros(self):
        args = ("--checkpoint", self.checkpoint)
        comparar.postprocesar("nms", "nms", *args)
        comparar.postprocesar("nms", "nms", *args)
        self.assertEqual(self.ejecutar.call_count, 1)
        self.guardar_checkpoint(self.checkpoint, self.config, revision=2)
        comparar.postprocesar("nms", "nms", *args)
        self.assertEqual(self.ejecutar.call_count, 2)
        parser = postprocesado.crear_parser()
        parsear = parser.parse_args

        def parsear_con_otros_umbrales(argumentos):
            opciones = parsear(argumentos)
            opciones.umbrales_score = [0.1, 0.2]
            return opciones

        with patch.object(parser, "parse_args", side_effect=parsear_con_otros_umbrales), patch.object(
            comparar, "crear_parser_postprocesado", return_value=parser,
        ):
            comparar.postprocesar("nms", "nms", *args)
        self.assertEqual(self.ejecutar.call_count, 3)

        comparar.postprocesar("tta", "tta", *args, "--vistas", "original", "hflip")
        comparar.postprocesar("tta", "tta", *args, "--vistas", "original", "vflip")
        self.assertEqual(self.ejecutar.call_count, 5)
        self.guardar_checkpoint(self.checkpoint, self.config, revision=3)
        comparar.postprocesar("tta", "tta", *args, "--vistas", "original", "vflip")
        self.assertEqual(self.ejecutar.call_count, 6)

    def test_ensemble_invalida_por_contenido_pesos_e_iou(self):
        rutas = [self.raiz / nombre for nombre in ("a.json", "b.json")]
        for ruta in rutas:
            escribir_json(ruta, [])
        args = ("--particion", "validation", "--preds", *rutas)
        comparar.postprocesar("ensemble", "fusion", *args)
        comparar.postprocesar("ensemble", "fusion", *args, "--pesos", "1", "1")
        self.assertEqual(self.ejecutar.call_count, 1)
        escribir_json(rutas[1], [{"image_id": 9}])
        comparar.postprocesar("ensemble", "fusion", *args)
        self.assertEqual(self.ejecutar.call_count, 2)
        comparar.postprocesar("ensemble", "fusion", *args, "--pesos", "1", "2")
        comparar.postprocesar("ensemble", "fusion", *args, "--pesos", "1", "2", "--iou-wbf", "0.65")
        self.assertEqual(self.ejecutar.call_count, 4)

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
            escribir_json(self.raiz / destinos[0] / "historial.json", [{"epoca": 1}])
            with self.assertRaisesRegex(RuntimeError, "entrenamiento completo"):
                comparar.main()
            self.assertEqual(self.ejecutar.call_count, 16)


if __name__ == "__main__":
    unittest.main()
