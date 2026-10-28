"""
Entrena el modelo de la fase 1 desde la línea de comandos.

Flujo: leer datos -> validar columnas -> descartar filas sin objetivo -> particionar
-> entrenar baseline y modelo -> evaluar -> guardar models/<version>/ con modelo.joblib,
metrics.json y metadata.json. La simulación opcional de faltantes está desactivada
por defecto: no sustituye faltantes naturales ni autorización docente.

Uso (desde fase-2/):
    python train.py --data data/flight_data_2024_sample.csv --version v1
    python train.py --data data/flight_data_2024_sample.csv --version v2 --particion fecha
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from pipeline import (
    COLUMNAS_PROHIBIDAS,
    OBJETIVO_POR_DEFECTO,
    PREDICTORAS,
    construir_modelo,
    validar_columnas,
)

RANDOM_STATE = 42
COLUMNA_FECHA = "fl_date"

JUSTIFICACION_PARTICION = {
    "aleatoria": (
        "train_test_split por filas, igual que la fase 1. Los días de prueba también "
        "tienen vuelos en entrenamiento, así que las métricas describen vuelos de días "
        "ya representados, no días nuevos."
    ),
    "fecha": (
        "GroupShuffleSplit por fl_date: ningún día de prueba aparece en entrenamiento. "
        "Como los retrasos se propagan dentro de un mismo día, evalúa la generalización "
        "a días no vistos del mismo periodo (no es una evaluación hacia el futuro)."
    ),
}

# Aeropuertos poco frecuentes pueden aparecer solo en prueba. El OneHotEncoder usa
# handle_unknown="ignore" justamente para eso, así que el aviso es esperado y solo
# ensuciaría la salida
warnings.filterwarnings("ignore", message="Found unknown categories")


def parsear_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de predicción de retrasos de llegada."
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de vuelos")
    parser.add_argument(
        "--version", required=True,
        help="Versión del modelo (p. ej. v1); se guarda en <modelos-dir>/<version>/",
    )
    parser.add_argument(
        "--modelos-dir", default="models",
        help="Carpeta raíz de los modelos versionados (por defecto: models)",
    )
    parser.add_argument(
        "--sobrescribir", action="store_true",
        help="Permite reemplazar una versión existente",
    )
    parser.add_argument(
        "--objetivo", default=OBJETIVO_POR_DEFECTO,
        help=f"Variable objetivo numérica (por defecto: {OBJETIVO_POR_DEFECTO})",
    )
    parser.add_argument(
        "--particion", choices=["aleatoria", "fecha"], default="aleatoria",
        help="aleatoria: train_test_split por filas (igual que la fase 1). "
             "fecha: agrupa por fl_date para que ningún día de prueba aparezca "
             "en entrenamiento",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fracción de prueba. Con --particion fecha es la fracción de días, "
             "no de filas (por defecto: 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help=f"Semilla para el muestreo y la partición (por defecto: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--fraccion-faltantes", type=float, default=0.0,
        help="Fracción sintética de NaN en crs_elapsed_time para probar imputación; "
             "no son faltantes naturales. Requiere autorización para presentarlos "
             "como requisito del curso (por defecto: 0)",
    )
    args = parser.parse_args(argv)

    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    if not 0 <= args.fraccion_faltantes < 1:
        parser.error("--fraccion-faltantes debe estar en [0, 1)")
    # La versión se usa como nombre de carpeta: sin separadores de ruta
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("--version solo admite letras, números, '.', '_' y '-' (p. ej. v1)")
    return args


def validar_objetivo(df, objetivo):
    """El objetivo debe existir, ser numérico y no ser una predictora ni una
    columna con fuga (p. ej. --objetivo dep_delay se rechaza)."""
    if objetivo in PREDICTORAS:
        raise ValueError(f"El objetivo '{objetivo}' es una de las predictoras.")
    if objetivo in COLUMNAS_PROHIBIDAS:
        raise ValueError(
            f"El objetivo '{objetivo}' es una columna posterior al vuelo prohibida."
        )
    validar_columnas(df, [objetivo])
    if not pd.api.types.is_numeric_dtype(df[objetivo]):
        raise ValueError(f"El objetivo '{objetivo}' debe ser numérico (regresión).")


def inducir_faltantes(df, fraccion, random_state):
    """
    Inyecta NaN en crs_elapsed_time, igual que la sección 2.7 del notebook.

    Ninguna predictora del dataset original tiene faltantes (0.00 % en 7 M filas);
    la fase 1 los indujo de forma declarada para ejercitar la imputación. Se aplica
    sobre el DataFrame completo, antes de descartar filas, con la misma semilla, para
    afectar exactamente las mismas filas que en la fase 1.
    """
    df = df.copy()
    if fraccion > 0:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(df.index, size=int(len(df) * fraccion), replace=False)
        df.loc[idx, "crs_elapsed_time"] = np.nan
    return df


def particionar(df, objetivo, particion, test_size, random_state):
    X = df[PREDICTORAS]
    y = df[objetivo]

    if particion == "aleatoria":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Los retrasos se propagan dentro de un mismo día (clima, congestión,
        # aeronaves encadenadas). Agrupar por fecha evalúa la generalización a
        # días no vistos en lugar de a vuelos de días ya representados
        separador = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        idx_train, idx_test = next(separador.split(X, y, groups=df[COLUMNA_FECHA]))
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    return X_train, X_test, y_train, y_test


def resumir_dias(df, X_train, X_test):
    """Cuántos días de prueba también tienen vuelos en entrenamiento."""
    if COLUMNA_FECHA not in df.columns:
        return None
    dias_train = set(df.loc[X_train.index, COLUMNA_FECHA])
    dias_test = set(df.loc[X_test.index, COLUMNA_FECHA])
    return {
        "n_dias_train": len(dias_train),
        "n_dias_test": len(dias_test),
        "dias_test_en_train": len(dias_test & dias_train),
    }


def calcular_metricas(y_real, y_pred):
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(root_mean_squared_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }


def mejora_pct(base, modelo):
    """Reducción porcentual del error frente al baseline (negativo = empeora)."""
    return (base - modelo) / base * 100


def evaluar(estimador, X_train, X_test, y_train, y_test):
    return {
        "train": calcular_metricas(y_train, estimador.predict(X_train)),
        "test": calcular_metricas(y_test, estimador.predict(X_test)),
    }


def imprimir_metricas(metricas, dias):
    print("\n=== Métricas ===")
    print(f"{'':10}{'conjunto':>10}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for nombre in ("baseline", "modelo"):
        for conjunto in ("train", "test"):
            m = metricas[nombre][conjunto]
            print(f"{nombre:10}{conjunto:>10}{m['mae']:>10.2f}"
                  f"{m['rmse']:>10.2f}{m['r2']:>10.4f}")

    print(f"\nMejora MAE en test frente al baseline:  {metricas['mejora_mae_pct']:+.2f} %")
    print(f"Mejora RMSE en test frente al baseline: {metricas['mejora_rmse_pct']:+.2f} %")

    if dias is not None:
        print(f"\nDías en entrenamiento: {dias['n_dias_train']}  |  "
              f"días en prueba: {dias['n_dias_test']}  |  "
              f"días de prueba también en entrenamiento: {dias['dias_test_en_train']}")


def hash_archivo(ruta):
    """SHA-256 del CSV: identifica exactamente con qué datos se entrenó."""
    sha = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def hash_codigo():
    """Identifica las fuentes principales cuando no hay metadatos de Git en runtime."""
    sha = hashlib.sha256()
    for nombre in ("pipeline.py", "train.py"):
        ruta = Path(__file__).with_name(nombre)
        sha.update(nombre.encode("utf-8"))
        with ruta.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                sha.update(bloque)
    return sha.hexdigest()


def construir_metadata(args, df, df_modelar, X_train, X_test, dias):
    return {
        "version": args.version,
        "fecha_entrenamiento": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tipo_problema": "regresion",
        "algoritmo": "LinearRegression",
        "parametros_modelo": {"fit_intercept": True},
        "baseline": "DummyRegressor(strategy='median')",
        "objetivo": args.objetivo,
        "predictoras": PREDICTORAS,
        "tipos_predictoras": {
            "op_unique_carrier": "categorica",
            "origin": "categorica",
            "dest": "categorica",
            "crs_dep_time": "numerica",
            "crs_arr_time": "numerica",
            "crs_elapsed_time": "numerica",
            "distance": "numerica",
            "month": "numerica",
            "day_of_week": "numerica",
        },
        "codigo": {"sha256": hash_codigo()},
        "particion": {
            "tipo": args.particion,
            "justificacion": JUSTIFICACION_PARTICION[args.particion],
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
        "datos": {
            # Solo el nombre: las rutas absolutas cambian en cada computador
            "archivo": Path(args.data).name,
            "fuente": (
                "Kaggle — Flight Delay Dataset 2024 "
                "(https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)"
            ),
            "sha256": hash_archivo(args.data),
            "filas_leidas": len(df),
            "filas_con_objetivo": len(df_modelar),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dias": dias,
            "alcance": "Solo vuelos con llegada observada; cancelados y desviados "
                       "se excluyen porque no tienen objetivo.",
        },
        "faltantes_inducidos": {
            "columna": "crs_elapsed_time",
            "fraccion": args.fraccion_faltantes,
            "nota": (
                "Las predictoras originales no tienen faltantes. Una fracción mayor "
                "que cero es una simulación sintética, no faltantes naturales. El equipo "
                "cuenta con autorización escrita para usar esta simulación documentada "
                "como alternativa al requisito."
            ),
        },
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def guardar_artefactos(directorio, modelo, metricas, metadata):
    directorio.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, directorio / "modelo.joblib")
    for nombre, contenido in (("metrics.json", metricas), ("metadata.json", metadata)):
        with open(directorio / nombre, "w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, indent=2, ensure_ascii=False)
            archivo.write("\n")


def main(argv=None):
    args = parsear_argumentos(argv)

    # Se comprueba antes de entrenar: una versión publicada no debe cambiar en
    # silencio, porque sus métricas y metadatos dejarían de corresponder
    directorio_version = Path(args.modelos_dir) / args.version
    if directorio_version.exists() and not args.sobrescribir:
        sys.exit(f"Error: la versión '{args.version}' ya existe en {directorio_version}. "
                 f"Use otra versión o --sobrescribir.")

    if not Path(args.data).is_file():
        sys.exit(f"Error: no se encontró el archivo de datos '{args.data}'. "
                 f"Descárguelo de Kaggle y ubíquelo en fase-2/data/ (ver README).")

    df = pd.read_csv(args.data)
    try:
        validar_columnas(df, PREDICTORAS)
        validar_objetivo(df, args.objetivo)
        if args.particion == "fecha":
            validar_columnas(df, [COLUMNA_FECHA])
    except ValueError as error:
        sys.exit(f"Error: {error}")

    df = inducir_faltantes(df, args.fraccion_faltantes, args.random_state)

    # Los vuelos cancelados o desviados no tienen llegada observada: no se imputa
    # el objetivo, se descartan (el modelo solo aplica a vuelos que aterrizan)
    df_modelar = df.dropna(subset=[args.objetivo])
    print(f"Filas leídas: {len(df):,}  |  filas con objetivo: {len(df_modelar):,}")

    X_train, X_test, y_train, y_test = particionar(
        df_modelar, args.objetivo, args.particion, args.test_size, args.random_state
    )
    print(f"Partición '{args.particion}': train {len(X_train):,} filas, "
          f"test {len(X_test):,} filas")

    # Referencia mínima: la mediana del entrenamiento es el predictor constante
    # que minimiza el error absoluto, alineado con el MAE como métrica principal
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    modelo = construir_modelo().fit(X_train, y_train)

    metricas = {
        "baseline": evaluar(baseline, X_train, X_test, y_train, y_test),
        "modelo": evaluar(modelo, X_train, X_test, y_train, y_test),
    }
    metricas["mejora_mae_pct"] = mejora_pct(
        metricas["baseline"]["test"]["mae"], metricas["modelo"]["test"]["mae"]
    )
    metricas["mejora_rmse_pct"] = mejora_pct(
        metricas["baseline"]["test"]["rmse"], metricas["modelo"]["test"]["rmse"]
    )

    dias = resumir_dias(df_modelar, X_train, X_test)
    imprimir_metricas(metricas, dias)

    metadata = construir_metadata(args, df, df_modelar, X_train, X_test, dias)
    guardar_artefactos(directorio_version, modelo, metricas, metadata)
    print(f"\nModelo guardado en {directorio_version}: "
          f"modelo.joblib, metrics.json, metadata.json")


if __name__ == "__main__":
    main()
