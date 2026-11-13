"""API REST para consultar los modelos versionados de la fase 2."""

import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

ROOT_DIR = Path(__file__).resolve().parents[1]
FASE_2_DIR = ROOT_DIR / "fase-2"
if str(FASE_2_DIR) not in sys.path:
    sys.path.insert(0, str(FASE_2_DIR))

from pipeline import COLUMNAS_CATEGORICAS, PREDICTORAS  # noqa: E402
from predict import predecir, validar_datos_entrada  # noqa: E402

VERSION_MODELO = os.getenv("MODEL_VERSION", "v1")
DIRECTORIO_MODELOS = Path(
    os.getenv("MODELS_DIR", str(FASE_2_DIR / "models"))
)
if not DIRECTORIO_MODELOS.is_absolute():
    DIRECTORIO_MODELOS = ROOT_DIR / DIRECTORIO_MODELOS

COLUMNAS_NUMERICAS = set(PREDICTORAS) - set(COLUMNAS_CATEGORICAS)
app = FastAPI(title="API de predicción de retrasos de vuelos")


def _crear_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    faltantes = [columna for columna in PREDICTORAS if columna not in payload]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan variables predictoras obligatorias: {faltantes}",
        )

    adicionales = sorted(set(payload) - set(PREDICTORAS))
    if adicionales:
        raise HTTPException(
            status_code=422,
            detail=f"Columnas no permitidas en la solicitud: {adicionales}",
        )

    for columna in PREDICTORAS:
        valor = payload[columna]
        if valor is None:
            continue
        if columna in COLUMNAS_NUMERICAS:
            if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"La variable '{columna}' debe ser numérica o null.",
                )
            if not math.isfinite(valor):
                raise HTTPException(
                    status_code=422,
                    detail=f"La variable '{columna}' debe ser un número finito.",
                )
        elif not isinstance(valor, str):
            raise HTTPException(
                status_code=422,
                detail=f"La variable '{columna}' debe ser texto o null.",
            )

    datos = pd.DataFrame([{columna: payload[columna] for columna in PREDICTORAS}])
    for columna in COLUMNAS_NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="raise")
    return datos


def _cargar_modelo():
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", VERSION_MODELO):
        raise HTTPException(
            status_code=503,
            detail="La versión configurada del modelo no es válida.",
        )

    ruta_modelo = DIRECTORIO_MODELOS / VERSION_MODELO / "modelo.joblib"
    if not ruta_modelo.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"El modelo configurado '{VERSION_MODELO}' no está disponible.",
        )
    return joblib.load(ruta_modelo)


@app.get("/health")
def health():
    return {"estado": "en ejecución"}


@app.post("/predict")
def predict(payload: dict[str, Any], request: Request):
    datos = _crear_dataframe(payload)
    try:
        validar_datos_entrada(datos)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    modelo = _cargar_modelo()
    request.state.version_modelo = VERSION_MODELO
    resultado = predecir(modelo, datos)
    prediccion = float(resultado["prediccion_arr_delay"].iloc[0])
    request.state.prediccion = prediccion
    return {"prediccion": prediccion, "version": VERSION_MODELO}
