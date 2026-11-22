import sys
from pathlib import Path

import pytest

FASE3_DIR = Path(__file__).resolve().parents[1]
if str(FASE3_DIR) not in sys.path:
    sys.path.insert(0, str(FASE3_DIR))

# Registro de vuelo real tomado de fase-2/data/prediccion.csv (primera fila)
VUELO_VALIDO = {
    "op_unique_carrier": "MQ",
    "origin": "DFW",
    "dest": "RAP",
    "crs_dep_time": 1018,
    "crs_arr_time": 1149,
    "crs_elapsed_time": 151.0,
    "distance": 835.0,
    "month": 4,
    "day_of_week": 4,
}


@pytest.fixture
def vuelo_valido():
    return dict(VUELO_VALIDO)


@pytest.fixture
def ruta_registro(tmp_path, monkeypatch):
    """Cada prueba escribe en su propio registro, nunca en el de fase-4/."""
    ruta = tmp_path / "registro_solicitudes.csv"
    monkeypatch.setenv("REQUEST_LOG_PATH", str(ruta))
    return ruta


@pytest.fixture
def cliente(ruta_registro, monkeypatch):
    from fastapi.testclient import TestClient

    import api

    # Modelo pequeño ya versionado en fase-2/models/v1 (17 KB)
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.delenv("MODELS_DIR", raising=False)
    return TestClient(api.app)
