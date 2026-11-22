"""
Cliente de prueba de la API (solo biblioteca estándar).

Consulta /health, envía dos vuelos reales de fase-2/data/prediccion.csv a /predict
y un tercero sin `distance` para comprobar el rechazo. Con --train también dispara
un entrenamiento. Es la misma secuencia del ensayo de la fase 4: sobre un registro
vacío deja 3 filas (2 x 200 y 1 x 422).

Uso (con la API en ejecución):
    python cliente_prueba.py
    python cliente_prueba.py --url http://localhost:8000 --train
"""

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RUTA_MUESTRA = Path(__file__).resolve().parents[1] / "fase-2" / "data" / "prediccion.csv"
COLUMNAS_NUMERICAS = {
    "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance", "month", "day_of_week",
}
COLUMNAS_ENTERAS = {"crs_dep_time", "crs_arr_time", "month", "day_of_week"}


def leer_vuelos(ruta, cantidad):
    """Convierte filas del CSV en JSON con los tipos del entrenamiento: número
    como número, categoría como texto y celda vacía como null."""
    vuelos = []
    with open(ruta, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            vuelo = {}
            for columna, valor in fila.items():
                if columna not in COLUMNAS_NUMERICAS:
                    vuelo[columna] = valor
                elif valor == "":
                    vuelo[columna] = None
                elif columna in COLUMNAS_ENTERAS:
                    vuelo[columna] = int(float(valor))
                else:
                    vuelo[columna] = float(valor)
            vuelos.append(vuelo)
            if len(vuelos) == cantidad:
                break
    return vuelos


def solicitar(metodo, url, cuerpo=None, timeout=300):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(
        url, data=datos, method=metodo, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            return respuesta.status, json.loads(respuesta.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def mostrar(titulo, metodo, url, cuerpo, estado, respuesta):
    print(f"\n### {titulo}")
    print(f"{metodo} {url}")
    if cuerpo is not None:
        print(f"Solicitud: {json.dumps(cuerpo, ensure_ascii=False)}")
    print(f"HTTP {estado}: {json.dumps(respuesta, ensure_ascii=False)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de retrasos.")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--train", action="store_true", help="Ejecutar también POST /train")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")

    try:
        estado, respuesta = solicitar("GET", f"{base}/health")
    except urllib.error.URLError as error:
        sys.exit(f"No se pudo conectar con {base}: {error.reason}. ¿Está la API en ejecución?")
    mostrar("Estado del servicio", "GET", f"{base}/health", None, estado, respuesta)

    vuelos = leer_vuelos(RUTA_MUESTRA, 2)
    incompleto = {k: v for k, v in vuelos[0].items() if k != "distance"}
    casos = [
        ("Predicción válida 1", vuelos[0]),
        ("Predicción válida 2", vuelos[1]),
        ("Predicción inválida (falta distance)", incompleto),
    ]
    for titulo, vuelo in casos:
        estado, respuesta = solicitar("POST", f"{base}/predict", vuelo)
        mostrar(titulo, "POST", f"{base}/predict", vuelo, estado, respuesta)

    if args.train:
        estado, respuesta = solicitar("POST", f"{base}/train")
        mostrar("Entrenamiento", "POST", f"{base}/train", None, estado, respuesta)


if __name__ == "__main__":
    main()
