"""
Pipeline de preprocesamiento y modelo de la fase 1, extraído a un módulo importable.

En la fase 1 la clase IngenieriaCaracteristicas estaba definida dentro del notebook,
es decir, en __main__. joblib serializa las clases por referencia (módulo + nombre),
así que el modelo guardado no podía cargarse desde un proceso distinto al notebook.
Al vivir aquí, el artefacto referencia `pipeline.IngenieriaCaracteristicas` y puede
cargarse desde train.py, predict.py o cualquier script que importe este módulo.

La lógica es la misma de la fase 1: no se cambia el modelo, solo cómo se empaqueta.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Las 9 predictoras aprobadas: todas se conocen antes del despegue
PREDICTORAS = [
    "op_unique_carrier", "origin", "dest",
    "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
    "distance", "month", "day_of_week",
]

OBJETIVO_POR_DEFECTO = "arr_delay"

# Columnas que solo se conocen durante o después del vuelo: usarlas como
# predictoras sería fuga de información
COLUMNAS_PROHIBIDAS = {
    "dep_time", "dep_delay", "taxi_out", "taxi_in", "wheels_off", "wheels_on",
    "air_time", "actual_elapsed_time", "arr_time", "arr_delay_new", "dep_delay_new",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay",
    "late_aircraft_delay", "cancelled", "cancellation_code", "diverted",
}

COLUMNAS_HORAS = ["crs_dep_time", "crs_arr_time"]

# Variables numéricas después de la ingeniería de características
COLUMNAS_NUMERICAS = [
    "crs_elapsed_time",
    "distance",
    "crs_dep_time_sin",
    "crs_dep_time_cos",
    "crs_arr_time_sin",
    "crs_arr_time_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

COLUMNAS_CATEGORICAS = ["op_unique_carrier", "origin", "dest"]

# Salvaguarda: si alguien agrega una columna prohibida a PREDICTORAS, el módulo
# falla al importarse en lugar de entrenar un modelo con fuga
_fuga = set(PREDICTORAS) & COLUMNAS_PROHIBIDAS
if _fuga:
    raise RuntimeError(f"Columnas con fuga de información en PREDICTORAS: {sorted(_fuga)}")


class IngenieriaCaracteristicas(BaseEstimator, TransformerMixin):
    """
    Genera características cíclicas para variables temporales.

    Las horas (hhmm), el mes y el día de la semana son circulares: 23:59 está a un
    minuto de 00:00 y diciembre es vecino de enero. Codificarlos con seno y coseno
    conserva esa cercanía, que un valor numérico lineal rompería.
    """

    def __init__(
        self,
        columnas_horas,
        columna_mes="month",
        columna_dia="day_of_week"
    ):
        self.columnas_horas = columnas_horas
        self.columna_mes = columna_mes
        self.columna_dia = columna_dia

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_temp = X.copy()

        for col in self.columnas_horas:
            horas = X_temp[col] // 100
            minutos = X_temp[col] % 100
            minutos_dia = horas * 60 + minutos

            X_temp[f"{col}_sin"] = np.sin(2 * np.pi * minutos_dia / 1440)
            X_temp[f"{col}_cos"] = np.cos(2 * np.pi * minutos_dia / 1440)

            X_temp.drop(columns=[col], inplace=True)

        X_temp["month_sin"] = np.sin(2 * np.pi * (X_temp[self.columna_mes] - 1) / 12)
        X_temp["month_cos"] = np.cos(2 * np.pi * (X_temp[self.columna_mes] - 1) / 12)

        X_temp["day_of_week_sin"] = np.sin(2 * np.pi * X_temp[self.columna_dia] / 7)
        X_temp["day_of_week_cos"] = np.cos(2 * np.pi * X_temp[self.columna_dia] / 7)

        X_temp.drop(columns=[self.columna_mes, self.columna_dia], inplace=True)

        return X_temp


def construir_preprocesador():
    """ColumnTransformer de la fase 1: mediana + escalado | One-Hot."""
    pipeline_numerico = Pipeline(
        steps=[
            # Mediana: robusta frente a la cola larga de distance y crs_elapsed_time
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )

    # handle_unknown="ignore": un aeropuerto o aerolínea no vista en entrenamiento
    # no rompe la predicción; drop="first" evita la colinealidad perfecta en la
    # regresión lineal
    pipeline_categorico = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, COLUMNAS_NUMERICAS),
            ("categoricas", pipeline_categorico, COLUMNAS_CATEGORICAS),
        ]
    )


def construir_modelo():
    """
    Pipeline completo sin entrenar: ingeniería -> preprocesamiento -> regresión.

    Todo el preprocesamiento va dentro del Pipeline para que .fit() aprenda la
    mediana, el escalado y las categorías únicamente con los datos de entrenamiento.
    """
    return Pipeline(
        steps=[
            ("ingenieria", IngenieriaCaracteristicas(columnas_horas=COLUMNAS_HORAS)),
            ("preprocesamiento", construir_preprocesador()),
            ("regresion", LinearRegression()),
        ]
    )


def validar_columnas(df, requeridas):
    """Lanza ValueError indicando qué columnas requeridas no están en df."""
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en los datos de entrada: {faltantes}. "
            f"Se esperan las predictoras: {PREDICTORAS}"
        )
