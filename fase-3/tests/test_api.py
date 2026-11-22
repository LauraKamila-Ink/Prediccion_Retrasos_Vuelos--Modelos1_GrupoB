import csv
import hashlib
import shutil

import pandas as pd
import pytest

import api

# --- Las tres pruebas exigidas por la especificación -------------------------


def test_health_responde_200(cliente):
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "en ejecución"
    assert cuerpo["version_configurada"] == "v1"
    assert cuerpo["modelo_disponible"] is True


def test_predict_valido_devuelve_prediccion_y_version(cliente, vuelo_valido):
    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"prediccion", "version"}
    assert isinstance(cuerpo["prediccion"], float)
    assert cuerpo["version"] == "v1"


def test_predict_sin_campo_obligatorio_se_rechaza(cliente, vuelo_valido):
    del vuelo_valido["distance"]

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 422
    assert "distance" in respuesta.json()["detail"]


# --- Validación adicional ----------------------------------------------------


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("distance", "835"),          # número enviado como texto
        ("origin", 123),              # categoría enviada como número
        ("origin", None),             # sin regla de imputación para categorías
        ("crs_dep_time", 1490),       # 90 minutos no es una hora hhmm
        ("month", 13),
        ("day_of_week", 0),
        ("distance", -5),
    ],
)
def test_predict_tipo_o_rango_invalido_se_rechaza(cliente, vuelo_valido, campo, valor):
    vuelo_valido[campo] = valor

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 422
    assert campo in respuesta.json()["detail"]


def test_predict_con_objetivo_o_columna_extra_se_rechaza(cliente, vuelo_valido):
    vuelo_valido["arr_delay"] = 12.0

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 422
    assert "arr_delay" in respuesta.json()["detail"]


def test_predict_null_numerico_usa_imputacion_de_fase2(cliente, vuelo_valido):
    vuelo_valido["crs_elapsed_time"] = None

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 200
    assert isinstance(respuesta.json()["prediccion"], float)


def test_predict_coincide_con_predict_py(cliente, vuelo_valido):
    """La API aplica el mismo modelo y preprocesamiento que la fase 2."""
    import joblib
    from predict import predecir

    modelo = joblib.load(api.FASE_2_DIR / "models" / "v1" / "modelo.joblib")
    esperado = predecir(modelo, pd.DataFrame([vuelo_valido]))["prediccion_arr_delay"][0]

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.json()["prediccion"] == pytest.approx(esperado)


def test_version_seleccionable_por_configuracion(cliente, vuelo_valido, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 200
    assert respuesta.json()["version"] == "v2"


def test_version_no_disponible_devuelve_503(cliente, vuelo_valido, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v999")

    respuesta = cliente.post("/predict", json=vuelo_valido)

    assert respuesta.status_code == 503
    assert "v999" in respuesta.json()["detail"]
    assert cliente.get("/health").json()["modelo_disponible"] is False


# --- Registro de la fase 4 ---------------------------------------------------


def leer_registro(ruta):
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def test_registro_guarda_una_fila_por_intento(cliente, vuelo_valido, ruta_registro):
    """Dos válidas y una sin campo obligatorio: tres filas, una con 422."""
    cliente.post("/predict", json=vuelo_valido)
    cliente.post("/predict", json={**vuelo_valido, "month": 7})
    incompleto = dict(vuelo_valido)
    del incompleto["distance"]
    cliente.post("/predict", json=incompleto)
    cliente.get("/health")  # no es /predict: no se registra

    filas = leer_registro(ruta_registro)

    assert [fila["estado_http"] for fila in filas] == ["200", "200", "422"]
    assert [fila["version"] for fila in filas] == ["v1", "v1", ""]
    assert filas[0]["prediccion"] and filas[1]["prediccion"]
    assert filas[2]["prediccion"] == ""


def test_registro_incluye_json_malformado(cliente, ruta_registro):
    respuesta = cliente.post(
        "/predict", content="{no es json", headers={"Content-Type": "application/json"}
    )

    assert respuesta.status_code == 422
    assert [fila["estado_http"] for fila in leer_registro(ruta_registro)] == ["422"]


# --- /train ------------------------------------------------------------------


def crear_datos_entrenamiento(ruta):
    filas = []
    aerolineas = ["AA", "DL", "UA"]
    aeropuertos = ["JFK", "LAX", "ORD", "DFW"]
    for indice in range(40):
        filas.append(
            {
                "op_unique_carrier": aerolineas[indice % 3],
                "origin": aeropuertos[indice % 4],
                "dest": aeropuertos[(indice + 1) % 4],
                "crs_dep_time": 600 + (indice % 12) * 100,
                "crs_arr_time": 900 + (indice % 12) * 100,
                "crs_elapsed_time": 90.0 + (indice % 8) * 30,
                "distance": 300.0 + (indice % 10) * 125,
                "month": indice % 12 + 1,
                "day_of_week": indice % 7 + 1,
                "arr_delay": float((indice * 7) % 45 - 10),
            }
        )
    pd.DataFrame(filas).to_csv(ruta, index=False)


def hash_archivo(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def test_train_crea_version_nueva_sin_modificar_las_anteriores(
    cliente, tmp_path, monkeypatch, vuelo_valido
):
    modelos = tmp_path / "models"
    for version in ("v1", "v2"):
        shutil.copytree(api.FASE_2_DIR / "models" / version, modelos / version)
    hashes_antes = {
        version: hash_archivo(modelos / version / "modelo.joblib")
        for version in ("v1", "v2")
    }
    datos = tmp_path / "vuelos.csv"
    crear_datos_entrenamiento(datos)
    monkeypatch.setenv("MODELS_DIR", str(modelos))
    monkeypatch.setenv("TRAIN_DATA_PATH", str(datos))

    primera = cliente.post("/train")
    segunda = cliente.post("/train")

    assert primera.status_code == 201
    assert primera.json()["version"] == "v3"
    assert segunda.json()["version"] == "v4"
    for version in ("v3", "v4"):
        assert {p.name for p in (modelos / version).iterdir()} == {
            "modelo.joblib", "metrics.json", "metadata.json"
        }
    for version, hash_antes in hashes_antes.items():
        assert hash_archivo(modelos / version / "modelo.joblib") == hash_antes

    # Entrenar no cambia la versión activa
    assert primera.json()["version_activa"] == "v1"
    assert cliente.post("/predict", json=vuelo_valido).json()["version"] == "v1"


def test_train_sin_datos_devuelve_503(cliente, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("TRAIN_DATA_PATH", str(tmp_path / "no_existe.csv"))

    respuesta = cliente.post("/train")

    assert respuesta.status_code == 503
    assert "TRAIN_DATA_PATH" in respuesta.json()["detail"]
    assert not (tmp_path / "models").exists()


def test_siguiente_version_ignora_nombres_no_numericos(tmp_path):
    for nombre in ("v1", "v10", "v-docker", "v2b"):
        (tmp_path / nombre).mkdir()

    assert api.siguiente_version(tmp_path) == "v11"
    assert api.siguiente_version(tmp_path / "vacia") == "v1"
