import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline import PREDICTORAS, construir_modelo
from predict import main as predict_main
from predict import predecir, validar_datos_entrada
from train import main as train_main

FASE2_DIR = Path(__file__).resolve().parents[1]
VERSIONES_ENTREGA = ("v1", "v2")


def crear_datos_prueba():
    filas = []
    aerolineas = ["AA", "DL", "UA"]
    aeropuertos = ["JFK", "LAX", "ORD", "DFW"]
    for indice in range(30):
        filas.append(
            {
                "op_unique_carrier": aerolineas[indice % len(aerolineas)],
                "origin": aeropuertos[indice % len(aeropuertos)],
                "dest": aeropuertos[(indice + 1) % len(aeropuertos)],
                "crs_dep_time": 600 + (indice % 12) * 100,
                "crs_arr_time": 900 + (indice % 12) * 100,
                "crs_elapsed_time": 90.0 + (indice % 8) * 30,
                "distance": 300.0 + (indice % 10) * 125,
                "month": indice % 12 + 1,
                "day_of_week": indice % 7,
                "arr_delay": float((indice * 7) % 45 - 10),
            }
        )
    return pd.DataFrame(filas)


@pytest.fixture
def datos_csv(tmp_path):
    ruta = tmp_path / "vuelos.csv"
    crear_datos_prueba().to_csv(ruta, index=False)
    return ruta


@pytest.mark.parametrize("version", VERSIONES_ENTREGA)
def test_modelos_versionados_de_entrega_se_cargan(version):
    version_dir = FASE2_DIR / "models" / version
    modelo = joblib.load(version_dir / "modelo.joblib")
    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["version"] == version
    assert metadata["tipo_problema"] == "regresion"
    assert metadata["codigo"]["commit"]
    assert len(modelo.predict(crear_datos_prueba().loc[:1, PREDICTORAS])) == 2


def test_muestra_inferencia_sin_objetivo():
    ruta_muestra = FASE2_DIR / "data" / "prediccion.csv"
    muestra = pd.read_csv(ruta_muestra)

    assert muestra.columns.tolist() == PREDICTORAS
    assert "arr_delay" not in muestra.columns
    assert len(muestra) == 10


@pytest.fixture
def modelo_entrenado(tmp_path, datos_csv):
    directorio_modelos = tmp_path / "models"
    train_main(
        [
            "--data",
            str(datos_csv),
            "--version",
            "v1",
            "--modelos-dir",
            str(directorio_modelos),
            "--fraccion-faltantes",
            "0",
        ]
    )
    return directorio_modelos


def test_entrenamiento_guarda_modelo_metricas_y_metadatos(modelo_entrenado):
    version_dir = modelo_entrenado / "v1"
    modelo = joblib.load(version_dir / "modelo.joblib")
    metricas = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))

    assert callable(modelo.predict)
    assert {"modelo", "baseline"} <= metricas.keys()
    assert metadata["version"] == "v1"
    assert metadata["objetivo"] == "arr_delay"
    assert metadata["codigo"]["sha256"]
    assert metadata["datos"]["n_train"] + metadata["datos"]["n_test"] == 30


def test_modelo_serializado_conserva_predicciones(tmp_path):
    datos = crear_datos_prueba()
    modelo = construir_modelo().fit(datos[PREDICTORAS], datos["arr_delay"])
    entrada = datos.loc[:4, PREDICTORAS]
    predicciones_antes = modelo.predict(entrada)
    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, ruta_modelo)

    predicciones_despues = joblib.load(ruta_modelo).predict(entrada)

    np.testing.assert_allclose(predicciones_despues, predicciones_antes)


def test_valores_faltantes_y_categorias_nuevas_se_procesan(modelo_entrenado):
    modelo = joblib.load(modelo_entrenado / "v1" / "modelo.joblib")
    entrada = crear_datos_prueba().loc[:3, list(reversed(PREDICTORAS))].copy()
    entrada.loc[0, "crs_elapsed_time"] = np.nan
    entrada.loc[1, "op_unique_carrier"] = "NUEVA"

    with pytest.warns(UserWarning, match="unknown categories"):
        resultado = predecir(modelo, entrada)

    assert len(resultado) == len(entrada)
    assert resultado.columns.tolist() == PREDICTORAS + ["prediccion_arr_delay"]
    assert resultado["prediccion_arr_delay"].notna().all()


def test_error_columnas_faltantes():
    datos = crear_datos_prueba().loc[:, PREDICTORAS].drop(columns=["distance"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validar_datos_entrada(datos)


def test_error_tipo_incompatible():
    datos = crear_datos_prueba().loc[:, PREDICTORAS].copy()
    datos["distance"] = datos["distance"].astype(str)

    with pytest.raises(ValueError, match="tipo incompatible"):
        validar_datos_entrada(datos)


@pytest.mark.parametrize("columna_fuga", ["arr_delay", "dep_delay"])
def test_rechazo_data_leakage(columna_fuga):
    datos = crear_datos_prueba().loc[:, PREDICTORAS].copy()
    datos[columna_fuga] = 15.0

    with pytest.raises(ValueError, match="Se rechazan columnas"):
        validar_datos_entrada(datos)


def test_ejecucion_cli_predict_genera_csv(tmp_path, modelo_entrenado):
    archivo_entrada = tmp_path / "entrada.csv"
    archivo_salida = tmp_path / "salida" / "predicciones.csv"
    crear_datos_prueba().loc[:3, PREDICTORAS].to_csv(archivo_entrada, index=False)

    predict_main(
        [
            "--data",
            str(archivo_entrada),
            "--version",
            "v1",
            "--modelos-dir",
            str(modelo_entrenado),
            "--output",
            str(archivo_salida),
        ]
    )

    resultado = pd.read_csv(archivo_salida)
    assert len(resultado) == 4
    assert "prediccion_arr_delay" in resultado.columns


def test_cli_con_columna_ausente_falla_sin_crear_salida(
    tmp_path, modelo_entrenado
):
    archivo_entrada = tmp_path / "entrada_invalida.csv"
    archivo_salida = tmp_path / "salida" / "predicciones.csv"
    crear_datos_prueba().loc[:3, PREDICTORAS].drop(columns=["distance"]).to_csv(
        archivo_entrada, index=False
    )

    with pytest.raises(SystemExit, match="Faltan columnas requeridas"):
        predict_main(
            [
                "--data",
                str(archivo_entrada),
                "--version",
                "v1",
                "--modelos-dir",
                str(modelo_entrenado),
                "--output",
                str(archivo_salida),
            ]
        )

    assert not archivo_salida.exists()


def test_versionamiento_conserva_versiones_y_rechaza_duplicados(
    datos_csv, modelo_entrenado
):
    ruta_v1 = modelo_entrenado / "v1" / "modelo.joblib"
    bytes_v1_antes = ruta_v1.read_bytes()

    train_main(
        [
            "--data",
            str(datos_csv),
            "--version",
            "v2",
            "--modelos-dir",
            str(modelo_entrenado),
            "--fraccion-faltantes",
            "0",
        ]
    )

    assert ruta_v1.is_file()
    assert joblib.load(ruta_v1) is not None
    assert joblib.load(modelo_entrenado / "v2" / "modelo.joblib") is not None
    with pytest.raises(SystemExit, match="ya existe"):
        train_main(
            [
                "--data",
                str(datos_csv),
                "--version",
                "v1",
                "--modelos-dir",
                str(modelo_entrenado),
                "--fraccion-faltantes",
                "0",
            ]
        )
    assert ruta_v1.read_bytes() == bytes_v1_antes
