"""
Registro persistente de las solicitudes a POST /predict (fase 4).

Cada llamada agrega una fila al CSV con fecha y hora (UTC), código HTTP, versión
del modelo y predicción. En un rechazo la predicción queda vacía y, si no se llegó
a usar un modelo, también la versión. El archivo se abre en modo de agregar para
conservar el historial entre ejecuciones de la API.
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COLUMNAS_REGISTRO = ["fecha_hora", "estado_http", "version", "prediccion"]

# uvicorn atiende las rutas síncronas en varios hilos: sin bloqueo, dos solicitudes
# simultáneas podrían intercalar sus filas o escribir el encabezado dos veces
_BLOQUEO_REGISTRO = threading.Lock()


def registrar_solicitud(
    ruta_registro: Path, estado_http: int, version: str = "", prediccion: Any = ""
) -> None:
    ruta_registro = Path(ruta_registro)
    ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    with _BLOQUEO_REGISTRO:
        escribir_encabezado = (
            not ruta_registro.exists() or ruta_registro.stat().st_size == 0
        )
        with ruta_registro.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_REGISTRO)
            if escribir_encabezado:
                escritor.writeheader()
            escritor.writerow(
                {
                    "fecha_hora": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "estado_http": estado_http,
                    "version": version,
                    "prediccion": prediccion,
                }
            )
