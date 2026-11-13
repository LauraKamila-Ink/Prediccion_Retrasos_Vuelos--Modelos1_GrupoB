"""Resume el total de solicitudes y el porcentaje de respuestas fallidas."""

import argparse
import csv
from pathlib import Path

RUTA_REGISTRO_PREDETERMINADA = (
    Path(__file__).resolve().parent / "registro_solicitudes.csv"
)
COLUMNAS_REQUERIDAS = {"fecha_hora", "estado_http", "version", "prediccion"}


def calcular_metricas(ruta_registro: Path) -> tuple[int, float | None]:
    with ruta_registro.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if not lector.fieldnames or not COLUMNAS_REQUERIDAS.issubset(lector.fieldnames):
            raise ValueError(
                f"El registro debe incluir las columnas: {sorted(COLUMNAS_REQUERIDAS)}"
            )

        total = 0
        fallidas = 0
        for numero_fila, fila in enumerate(lector, start=2):
            try:
                estado_http = int(fila["estado_http"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"El estado HTTP de la fila {numero_fila} no es un entero válido."
                ) from error
            total += 1
            if 400 <= estado_http <= 599:
                fallidas += 1

    porcentaje = (fallidas / total * 100) if total else None
    return total, porcentaje


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calcula métricas operativas del registro de predicciones."
    )
    parser.add_argument(
        "--registro",
        type=Path,
        default=RUTA_REGISTRO_PREDETERMINADA,
        help="Ruta al CSV (por defecto: fase-4/registro_solicitudes.csv)",
    )
    args = parser.parse_args(argv)
    if not args.registro.is_file():
        parser.error(f"No se encontró el registro '{args.registro}'.")

    total, porcentaje = calcular_metricas(args.registro)
    print(f"Total de solicitudes: {total}")
    if porcentaje is None:
        print("Porcentaje de solicitudes fallidas: sin solicitudes")
    else:
        porcentaje_texto = f"{porcentaje:.1f}".replace(".", ",")
        print(f"Porcentaje de solicitudes fallidas: {porcentaje_texto} %")


if __name__ == "__main__":
    main()
