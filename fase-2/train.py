"""
Entrena el modelo de la fase 1 desde la línea de comandos.

Flujo: leer datos -> validar columnas -> inducir faltantes (declarados) -> descartar
filas sin objetivo -> particionar -> entrenar baseline y modelo -> evaluar.

Uso (desde fase-2/):
    python train.py --data data/flight_data_2024_sample.csv
    python train.py --data data/flight_data_2024_sample.csv --particion fecha
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
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
        "--fraccion-faltantes", type=float, default=0.01,
        help="Fracción de NaN inducidos en crs_elapsed_time, como en la fase 1. "
             "Use 0 para desactivarlos (por defecto: 0.01)",
    )
    args = parser.parse_args(argv)

    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    if not 0 <= args.fraccion_faltantes < 1:
        parser.error("--fraccion-faltantes debe estar en [0, 1)")
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


def main(argv=None):
    args = parsear_argumentos(argv)

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

    imprimir_metricas(metricas, resumir_dias(df_modelar, X_train, X_test))


if __name__ == "__main__":
    main()
