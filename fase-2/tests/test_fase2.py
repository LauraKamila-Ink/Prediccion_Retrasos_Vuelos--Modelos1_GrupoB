from pathlib import Path

import joblib
import pandas as pd
import pytest

from pipeline import PREDICTORAS
from predict import main, predecir, validar_datos_entrada


FASE2_DIR = Path(__file__).resolve().parents[1]
MODELOS_DIR = FASE2_DIR / "models"
VERSION_MODELO = "v2" if (MODELOS_DIR / "v2" / "modelo.joblib").is_file() else "v1"
RUTA_MODELO = MODELOS_DIR / VERSION_MODELO / "modelo.joblib"


def crear_datos_prueba():
    return pd.DataFrame(
        {
            "op_unique_carrier": ["AA", "DL", "UA"],
            "origin": ["JFK", "LAX", "ORD"],
            "dest": ["LAX", "ORD", "JFK"],
            "crs_dep_time": [800, 1230, 1745],
            "crs_arr_time": [1100, 1630, 2100],
            "crs_elapsed_time": [300.0, 240.0, 135.0],
            "distance": [2475.0, 1744.0, 740.0],
            "month": [1, 6, 11],
            "day_of_week": [1, 3, 5],
        }
    )[PREDICTORAS]


def test_cargar_modelo():
    assert RUTA_MODELO.is_file()
    modelo = joblib.load(RUTA_MODELO)
    assert callable(modelo.predict)


def test_prediccion_estructura():
    modelo = joblib.load(RUTA_MODELO)
    resultado = predecir(modelo, crear_datos_prueba())

    assert "prediccion_arr_delay" in resultado.columns
    assert len(resultado) == 3
    assert resultado[PREDICTORAS].equals(crear_datos_prueba())


def test_error_columnas_faltantes():
    datos = crear_datos_prueba().drop(columns=["distance"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validar_datos_entrada(datos)


@pytest.mark.parametrize("columna_fuga", ["arr_delay", "dep_delay"])
def test_rechazo_data_leakage(columna_fuga):
    datos = crear_datos_prueba()
    datos[columna_fuga] = 15.0

    with pytest.raises(ValueError, match="Se rechazan columnas"):
        validar_datos_entrada(datos)


def test_ejecucion_cli_predict(tmp_path):
    archivo_entrada = tmp_path / "entrada.csv"
    archivo_salida = tmp_path / "salida" / "predicciones.csv"
    crear_datos_prueba().to_csv(archivo_entrada, index=False)

    main(
        [
            "--data",
            str(archivo_entrada),
            "--version",
            VERSION_MODELO,
            "--modelos-dir",
            str(MODELOS_DIR),
            "--output",
            str(archivo_salida),
        ]
    )

    assert archivo_salida.is_file()
    resultado = pd.read_csv(archivo_salida)
    assert len(resultado) == 3
    assert "prediccion_arr_delay" in resultado.columns
