"""
API REST para consultar y entrenar los modelos versionados de la fase 2.

Rutas:
    GET  /health   estado del servicio y de la versión configurada
    POST /predict  predicción del retraso de llegada para un vuelo
    POST /train    entrena y guarda una versión nueva sin tocar las anteriores

Configuración (variables de entorno; las rutas relativas se resuelven desde la raíz
del repositorio):
    MODEL_VERSION      versión que atiende /predict (por defecto: v1)
    MODELS_DIR         carpeta de versiones (por defecto: fase-2/models)
    TRAIN_DATA_PATH    CSV de entrenamiento de /train
                       (por defecto: fase-2/data/flight_data_2024_sample.csv)
    REQUEST_LOG_PATH   registro de solicitudes de la fase 4
                       (por defecto: fase-4/registro_solicitudes.csv)

Uso (desde fase-3/):
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

ROOT_DIR = Path(__file__).resolve().parents[1]
FASE_2_DIR = ROOT_DIR / "fase-2"
FASE_4_DIR = ROOT_DIR / "fase-4"
for _directorio in (FASE_2_DIR, FASE_4_DIR):
    if str(_directorio) not in sys.path:
        sys.path.insert(0, str(_directorio))

from pipeline import COLUMNAS_CATEGORICAS, PREDICTORAS  # noqa: E402
from predict import predecir, validar_datos_entrada  # noqa: E402
from registro import registrar_solicitud  # noqa: E402
from train import RANDOM_STATE, entrenar_version  # noqa: E402

COLUMNAS_NUMERICAS = set(PREDICTORAS) - set(COLUMNAS_CATEGORICAS)
COLUMNAS_HHMM = {"crs_dep_time", "crs_arr_time"}
# Rangos del calendario y de las horas programadas. Un valor fuera de rango no es
# un vuelo posible: se rechaza en lugar de dejar que el modelo extrapole
RANGOS = {"month": (1, 12), "day_of_week": (1, 7)}
PATRON_VERSION = r"[A-Za-z0-9][A-Za-z0-9._-]*"

app = FastAPI(title="API de predicción de retrasos de vuelos")

# Dos /train simultáneos podrían calcular el mismo identificador
_BLOQUEO_ENTRENAMIENTO = threading.Lock()


def _ruta_configurada(variable: str, por_defecto: Path) -> Path:
    ruta = Path(os.getenv(variable, str(por_defecto)))
    return ruta if ruta.is_absolute() else ROOT_DIR / ruta


def obtener_configuracion() -> dict[str, Any]:
    """Se lee en cada solicitud: cambiar MODEL_VERSION y reiniciar basta para
    cambiar de versión, y las pruebas pueden ajustarla con monkeypatch."""
    return {
        "version": os.getenv("MODEL_VERSION", "v1"),
        "directorio_modelos": _ruta_configurada("MODELS_DIR", FASE_2_DIR / "models"),
        "datos_entrenamiento": _ruta_configurada(
            "TRAIN_DATA_PATH", FASE_2_DIR / "data" / "flight_data_2024_sample.csv"
        ),
        "ruta_registro": _ruta_configurada(
            "REQUEST_LOG_PATH", FASE_4_DIR / "registro_solicitudes.csv"
        ),
    }


@app.middleware("http")
async def registrar_predicciones(request: Request, call_next):
    """Una fila por intento en /predict, también para los rechazos de validación,
    porque el middleware envuelve tanto la ruta como los manejadores de error."""
    if request.url.path != "/predict":
        return await call_next(request)

    ruta_registro = obtener_configuracion()["ruta_registro"]
    try:
        response = await call_next(request)
    except Exception:
        registrar_solicitud(
            ruta_registro,
            500,
            request.scope.get("fase3_version_modelo", ""),
            request.scope.get("fase3_prediccion", ""),
        )
        raise

    registrar_solicitud(
        ruta_registro,
        response.status_code,
        request.scope.get("fase3_version_modelo", ""),
        request.scope.get("fase3_prediccion", ""),
    )
    return response


def _error_validacion(mensaje: str) -> HTTPException:
    return HTTPException(status_code=422, detail=mensaje)


def _validar_numero(columna: str, valor: Any) -> None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise _error_validacion(f"La variable '{columna}' debe ser numérica o null.")
    try:
        numero = float(valor)
    except OverflowError as error:
        raise _error_validacion(
            f"La variable '{columna}' debe ser un número finito."
        ) from error
    if not math.isfinite(numero):
        raise _error_validacion(f"La variable '{columna}' debe ser un número finito.")

    if columna in COLUMNAS_HHMM:
        # Formato hhmm de BTS: 1430 es válido, 1490 no; 2400 representa medianoche
        if not numero.is_integer() or not 0 <= numero <= 2400 or numero % 100 >= 60:
            raise _error_validacion(
                f"La variable '{columna}' debe ser una hora hhmm válida (0 a 2400)."
            )
    elif columna in RANGOS:
        minimo, maximo = RANGOS[columna]
        if not numero.is_integer() or not minimo <= numero <= maximo:
            raise _error_validacion(
                f"La variable '{columna}' debe ser un entero entre {minimo} y {maximo}."
            )
    elif numero <= 0:
        raise _error_validacion(f"La variable '{columna}' debe ser mayor que cero.")


def _crear_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    faltantes = [columna for columna in PREDICTORAS if columna not in payload]
    if faltantes:
        raise _error_validacion(f"Faltan variables predictoras obligatorias: {faltantes}")

    adicionales = sorted(set(payload) - set(PREDICTORAS))
    if adicionales:
        raise _error_validacion(f"Columnas no permitidas en la solicitud: {adicionales}")

    for columna in PREDICTORAS:
        valor = payload[columna]
        if columna in COLUMNAS_NUMERICAS:
            # null es un faltante permitido: lo resuelve la imputación por mediana
            # guardada en el pipeline de la fase 2
            if valor is not None:
                _validar_numero(columna, valor)
        elif not isinstance(valor, str) or not valor.strip():
            # La fase 2 no define una regla para categorías faltantes: se rechazan
            raise _error_validacion(
                f"La variable '{columna}' debe ser un texto no vacío."
            )

    # Orden del esquema de entrenamiento, independiente del orden del JSON
    datos = pd.DataFrame([{columna: payload[columna] for columna in PREDICTORAS}])
    for columna in COLUMNAS_NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="raise").astype(float)
    return datos


def _ruta_modelo(config: dict[str, Any]) -> Path | None:
    """Ruta del modelo configurado, o None si la versión no es un nombre válido."""
    if not re.fullmatch(PATRON_VERSION, config["version"]):
        return None
    return config["directorio_modelos"] / config["version"] / "modelo.joblib"


def _cargar_modelo(config: dict[str, Any]):
    ruta_modelo = _ruta_modelo(config)
    if ruta_modelo is None:
        raise HTTPException(
            status_code=503,
            detail="La versión configurada del modelo no es válida.",
        )
    if not ruta_modelo.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"El modelo configurado '{config['version']}' no está disponible. "
                f"Seleccione una versión existente con la variable MODEL_VERSION."
            ),
        )
    return joblib.load(ruta_modelo)


def siguiente_version(directorio_modelos: Path) -> str:
    """v<N+1>, donde N es el mayor número entre las carpetas v<N> existentes."""
    numeros = [
        int(coincidencia.group(1))
        for carpeta in directorio_modelos.glob("v*")
        if carpeta.is_dir() and (coincidencia := re.fullmatch(r"v(\d+)", carpeta.name))
    ]
    return f"v{max(numeros, default=0) + 1}"


@app.get("/health")
def health():
    config = obtener_configuracion()
    ruta_modelo = _ruta_modelo(config)
    return {
        "estado": "en ejecución",
        "version_configurada": config["version"],
        "modelo_disponible": ruta_modelo is not None and ruta_modelo.is_file(),
    }


@app.post("/predict")
def predict(payload: dict[str, Any], request: Request):
    config = obtener_configuracion()
    datos = _crear_dataframe(payload)
    try:
        validar_datos_entrada(datos)
    except ValueError as error:
        raise _error_validacion(str(error)) from error

    modelo = _cargar_modelo(config)
    request.scope["fase3_version_modelo"] = config["version"]
    resultado = predecir(modelo, datos)
    prediccion = float(resultado["prediccion_arr_delay"].iloc[0])
    request.scope["fase3_prediccion"] = prediccion
    return {"prediccion": prediccion, "version": config["version"]}


@app.post("/train", status_code=201)
def train():
    """
    Entrena con los datos y la configuración de la fase 2 (objetivo arr_delay,
    partición aleatoria, semilla 42, sin faltantes inducidos) y guarda una versión
    nueva. No cambia la versión que atiende /predict: activarla es una decisión
    explícita (MODEL_VERSION), según la política de la fase 4.
    """
    config = obtener_configuracion()
    if not config["datos_entrenamiento"].is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No se encontraron los datos de entrenamiento "
                f"'{config['datos_entrenamiento'].name}'. Configure TRAIN_DATA_PATH "
                f"o descargue la muestra de Kaggle en fase-2/data/."
            ),
        )

    with _BLOQUEO_ENTRENAMIENTO:
        version = siguiente_version(config["directorio_modelos"])
        try:
            directorio, metricas, _ = entrenar_version(
                data=config["datos_entrenamiento"],
                version=version,
                modelos_dir=config["directorio_modelos"],
                random_state=RANDOM_STATE,
            )
        except ValueError as error:
            raise _error_validacion(f"Datos de entrenamiento no válidos: {error}") from error

    return {
        "version": version,
        "directorio": directorio.relative_to(ROOT_DIR).as_posix()
        if directorio.is_relative_to(ROOT_DIR)
        else directorio.name,
        "metricas_test": {
            "modelo": metricas["modelo"]["test"],
            "baseline": metricas["baseline"]["test"],
        },
        "version_activa": config["version"],
    }
