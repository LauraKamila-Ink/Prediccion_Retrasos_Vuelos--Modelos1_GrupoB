"""
Genera predicciones de retraso de llegada desde un CSV de vuelos.

Uso (desde fase-2/):
    python predict.py --data data/prediccion.csv --version v1
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

from pipeline import (
    COLUMNAS_PROHIBIDAS,
    COLUMNAS_CATEGORICAS,
    OBJETIVO_POR_DEFECTO,
    PREDICTORAS,
    validar_columnas,
)


def parsear_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Predice el retraso de llegada de vuelos con un modelo entrenado."
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--version", required=True,
        help="Versión del modelo (p. ej. v1) dentro de <modelos-dir>/",
    )
    parser.add_argument(
        "--modelos-dir", default="models",
        help="Carpeta raíz de los modelos versionados (por defecto: models)",
    )
    parser.add_argument(
        "--output", default="outputs/predicciones.csv",
        help="Ruta del CSV de salida (por defecto: outputs/predicciones.csv)",
    )
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("--version solo admite letras, números, '.', '_' y '-' (p. ej. v1)")
    return args


def validar_datos_entrada(df):
    """Exige exclusivamente las nueve predictoras aprobadas, en cualquier orden."""
    validar_columnas(df, PREDICTORAS)

    columnas_numericas = {
        "crs_dep_time",
        "crs_arr_time",
        "crs_elapsed_time",
        "distance",
        "month",
        "day_of_week",
    }
    columnas_categoricas = set(COLUMNAS_CATEGORICAS)
    for columna in columnas_numericas:
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise ValueError(f"La columna numérica '{columna}' tiene un tipo incompatible.")
    for columna in columnas_categoricas:
        if not (
            pd.api.types.is_object_dtype(df[columna])
            or pd.api.types.is_string_dtype(df[columna])
            or isinstance(df[columna].dtype, pd.CategoricalDtype)
        ):
            raise ValueError(
                f"La columna categórica '{columna}' tiene un tipo incompatible."
            )

    fugas = sorted(set(df.columns) & (COLUMNAS_PROHIBIDAS | {OBJETIVO_POR_DEFECTO}))
    if fugas:
        raise ValueError(
            f"Se rechazan columnas con información posterior al despegue o el objetivo: "
            f"{fugas}"
        )

    adicionales = [col for col in df.columns if col not in PREDICTORAS]
    if adicionales:
        raise ValueError(
            f"Columnas no permitidas en inferencia: {adicionales}. "
            f"Use únicamente estas predictoras: {PREDICTORAS}"
        )


def predecir(modelo, df):
    """Valida las entradas y devuelve las predictoras junto a la predicción."""
    validar_datos_entrada(df)
    X = df.loc[:, PREDICTORAS]
    resultado = X.copy()
    resultado["prediccion_arr_delay"] = modelo.predict(X)
    return resultado


def main(argv=None):
    args = parsear_argumentos(argv)
    ruta_datos = Path(args.data)
    ruta_modelo = Path(args.modelos_dir) / args.version / "modelo.joblib"

    if not ruta_datos.is_file():
        sys.exit(f"Error: no se encontró el archivo de entrada '{ruta_datos}'.")
    if not ruta_modelo.is_file():
        sys.exit(f"Error: no se encontró el modelo '{ruta_modelo}'.")

    df = pd.read_csv(ruta_datos)
    try:
        resultado = predecir(joblib.load(ruta_modelo), df)
    except ValueError as error:
        sys.exit(f"Error: {error}")

    ruta_salida = Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(ruta_salida, index=False)
    print(f"Predicciones guardadas en {ruta_salida} ({len(resultado)} filas).")


if __name__ == "__main__":
    main()
