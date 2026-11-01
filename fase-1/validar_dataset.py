"""
Validación del dataset contra los requisitos de la Fase 1.

Ejecutar ANTES de escribir cualquier código de preprocesamiento.

Uso:
    python validar_dataset.py --data ruta/al/flight_data_2024_sample.csv

Requisitos verificados (Especificación Técnica, sección 5):
    1. Problema de clasificación o regresión
    2. Variable objetivo claramente definida
    3. Al menos 1.000 observaciones
    4. Al menos 5 variables predictoras
    5. Valores faltantes entre 0.1% y 2% en al menos una variable predictora
    6. No ser datos de series temporales
    7. Procesable en computador personal
"""

import argparse
import sys

import pandas as pd

# --- Configuración: las 9 predictoras aprobadas ---
PREDICTORAS = [
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
    "month",
    "day_of_week",
]

OBJETIVO = "arr_delay"

# Columnas que NO pueden usarse como predictoras (se conocen durante/después del vuelo)
PROHIBIDAS = {
    "dep_time", "dep_delay", "taxi_out", "taxi_in", "wheels_off", "wheels_on",
    "air_time", "actual_elapsed_time", "arr_time", "arr_delay_new", "dep_delay_new",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay",
    "late_aircraft_delay", "cancelled", "cancellation_code", "diverted",
}

RANGO_NULOS = (0.1, 2.0)  # porcentaje mínimo y máximo exigido


def separador(titulo):
    print(f"\n{'=' * 70}")
    print(f"  {titulo}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta al CSV")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Leer solo N filas (útil para el archivo completo de 7M)")
    args = parser.parse_args()

    print(f"Cargando {args.data} ...")
    df = pd.read_csv(args.data, nrows=args.nrows, low_memory=False)

    # Normalizar nombres de columna por si vienen en mayúsculas
    df.columns = [c.strip().lower() for c in df.columns]

    resultados = {}

    # ------------------------------------------------------------------
    separador("1. DIMENSIONES")
    n_filas, n_cols = df.shape
    print(f"Filas:    {n_filas:,}")
    print(f"Columnas: {n_cols}")
    resultados["≥ 1.000 observaciones"] = n_filas >= 1000

    memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memoria:  {memoria_mb:.1f} MB")
    resultados["Procesable en PC personal"] = memoria_mb < 2000

    # ------------------------------------------------------------------
    separador("2. COLUMNAS DISPONIBLES")
    print(f"Total: {len(df.columns)}\n")
    for i, c in enumerate(sorted(df.columns), 1):
        print(f"  {i:2d}. {c:<30} {str(df[c].dtype)}")

    # ------------------------------------------------------------------
    separador("3. VARIABLE OBJETIVO")
    if OBJETIVO in df.columns:
        print(f"'{OBJETIVO}' encontrada.")
        print(f"  Tipo:    {df[OBJETIVO].dtype}")
        print(f"  Nulos:   {df[OBJETIVO].isna().sum():,} "
              f"({df[OBJETIVO].isna().mean() * 100:.3f}%)")
        print(f"  Media:   {df[OBJETIVO].mean():.2f} min")
        print(f"  Mediana: {df[OBJETIVO].median():.2f} min")
        print(f"  Min/Max: {df[OBJETIVO].min():.0f} / {df[OBJETIVO].max():.0f} min")
        resultados["Variable objetivo definida"] = True
        resultados["Problema de regresión"] = True
    else:
        print(f"NO se encontró '{OBJETIVO}'.")
        print("Columnas parecidas:",
              [c for c in df.columns if "delay" in c or "arr" in c])
        resultados["Variable objetivo definida"] = False
        resultados["Problema de regresión"] = False

    # ------------------------------------------------------------------
    separador("4. VARIABLES PREDICTORAS APROBADAS")
    presentes = [c for c in PREDICTORAS if c in df.columns]
    faltantes = [c for c in PREDICTORAS if c not in df.columns]

    print(f"Presentes ({len(presentes)}/{len(PREDICTORAS)}):")
    for c in presentes:
        print(f"  ✓ {c:<25} {str(df[c].dtype):<12} "
              f"{df[c].nunique():>6,} valores únicos")

    if faltantes:
        print(f"\nNO encontradas ({len(faltantes)}):")
        for c in faltantes:
            print(f"  ✗ {c}")
        print("\n  → Revisa el nombre real en la lista de la sección 2.")

    resultados["≥ 5 variables predictoras"] = len(presentes) >= 5

    # ------------------------------------------------------------------
    separador("5. VALORES FALTANTES  ← REQUISITO CRÍTICO")
    nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    nulos = nulos[nulos > 0]

    if nulos.empty:
        print("El dataset NO tiene valores faltantes en ninguna columna.")
    else:
        print(f"{'Columna':<30} {'% nulos':>10}   Estado")
        print("-" * 70)
        for col, pct in nulos.items():
            if col in PREDICTORAS:
                if RANGO_NULOS[0] <= pct <= RANGO_NULOS[1]:
                    estado = "★ PREDICTORA EN RANGO"
                elif pct < RANGO_NULOS[0]:
                    estado = "predictora, muy pocos"
                else:
                    estado = "predictora, excede 2%"
            elif col in PROHIBIDAS:
                estado = "(prohibida - no cuenta)"
            elif col == OBJETIVO:
                estado = "(objetivo - no cuenta)"
            else:
                estado = "(no usada)"
            print(f"{col:<30} {pct:>9.3f}%   {estado}")

    # Veredicto del requisito de nulos
    candidatas = [
        c for c in presentes
        if RANGO_NULOS[0] <= df[c].isna().mean() * 100 <= RANGO_NULOS[1]
    ]
    cumple_nulos = len(candidatas) > 0
    resultados[f"Nulos {RANGO_NULOS[0]}%–{RANGO_NULOS[1]}% en predictora"] = cumple_nulos

    print()
    if cumple_nulos:
        print(f"✓ CUMPLE. Variables utilizables para justificar la imputación:")
        for c in candidatas:
            print(f"    - {c}  ({df[c].isna().mean() * 100:.3f}%)")
    else:
        print("✗ NO CUMPLE. Ninguna predictora aprobada tiene nulos en el rango exigido.")
        print("\n  Opciones, en orden de preferencia:")
        print("  1. Buscar otra columna del dataset que sirva como predictora legítima")
        print("     (pre-vuelo) y tenga nulos en rango. Revisa la sección 2 y 5.")
        print("  2. Probar con una muestra distinta o con el archivo completo: la tasa")
        print("     de nulos puede variar entre la muestra de 10k y los 7M de filas.")
        print("  3. Escribir al profesor explicando que el autor pre-limpió el dataset")
        print("     y pedir autorización explícita para inducir faltantes documentados")
        print("     o para cambiar de dataset. NO inducirlos sin respuesta escrita.")

    # ------------------------------------------------------------------
    separador("6. COLUMNAS PROHIBIDAS PRESENTES (fuga de información)")
    prohibidas_presentes = sorted(PROHIBIDAS & set(df.columns))
    if prohibidas_presentes:
        print("Estas columnas existen en el dataset y deben EXCLUIRSE de X:")
        for c in prohibidas_presentes:
            print(f"  ✗ {c}")
        print("\n  → Documenta esta exclusión en el notebook (sección 3.6).")
    else:
        print("Ninguna columna prohibida presente.")

    # ------------------------------------------------------------------
    separador("7. VERIFICACIÓN: ¿ES SERIE TEMPORAL?")
    print("Este dataset se modela como problema tabular transversal:")
    print("cada fila es un vuelo independiente, no una secuencia autorregresiva.")
    print("'month' y 'day_of_week' se usan como variables cíclicas, no como índice")
    print("temporal, y no se predice el futuro a partir del pasado de la misma serie.")
    resultados["No es serie temporal"] = True

    # ------------------------------------------------------------------
    separador("VEREDICTO")
    ancho = max(len(k) for k in resultados)
    for req, ok in resultados.items():
        print(f"  {'✓' if ok else '✗'}  {req:<{ancho}}  {'CUMPLE' if ok else 'NO CUMPLE'}")

    todos = all(resultados.values())
    print()
    if todos:
        print("TODOS LOS REQUISITOS SE CUMPLEN. Pueden proceder con la Fase 1.")
    else:
        fallidos = [k for k, v in resultados.items() if not v]
        print(f"HAY {len(fallidos)} REQUISITO(S) SIN CUMPLIR:")
        for f in fallidos:
            print(f"  - {f}")
        print("\nResuélvanlos antes de escribir el pipeline.")

    return 0 if todos else 1


if __name__ == "__main__":
    sys.exit(main())
