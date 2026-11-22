import pytest

from registro import registrar_solicitud
from reporte import calcular_metricas, main

ENCABEZADO = "fecha_hora,estado_http,version,prediccion\n"


def escribir_registro(ruta, filas):
    ruta.write_text(ENCABEZADO + "".join(f"{fila}\n" for fila in filas), encoding="utf-8")
    return ruta


def test_dos_validas_y_una_invalida_da_3_solicitudes_y_33_3(tmp_path, capsys):
    """Ejemplo de comprobación de la especificación."""
    ruta = escribir_registro(
        tmp_path / "registro.csv",
        [
            "2026-11-06T10:00:00,200,v1,18.7",
            "2026-11-06T10:01:00,200,v1,19.2",
            "2026-11-06T10:02:00,422,,",
        ],
    )

    total, porcentaje = calcular_metricas(ruta)
    main(["--registro", str(ruta)])
    salida = capsys.readouterr().out

    assert total == 3
    assert porcentaje == pytest.approx(100 / 3)
    assert "Total de solicitudes: 3" in salida
    assert "33,3 %" in salida


@pytest.mark.parametrize("contenido", ["", ENCABEZADO])
def test_registro_vacio_muestra_total_0_y_sin_solicitudes(tmp_path, capsys, contenido):
    ruta = tmp_path / "registro.csv"
    ruta.write_text(contenido, encoding="utf-8")

    main(["--registro", str(ruta)])
    salida = capsys.readouterr().out

    assert calcular_metricas(ruta) == (0, None)
    assert "Total de solicitudes: 0" in salida
    assert "sin solicitudes" in salida


def test_cuenta_4xx_y_5xx_como_fallidas(tmp_path):
    ruta = escribir_registro(
        tmp_path / "registro.csv",
        [
            "2026-11-06T10:00:00,200,v1,18.7",
            "2026-11-06T10:01:00,400,,",
            "2026-11-06T10:02:00,503,,",
            "2026-11-06T10:03:00,302,,",
        ],
    )

    assert calcular_metricas(ruta) == (4, 50.0)


def test_registro_escrito_por_la_api_se_puede_resumir(tmp_path):
    """registro.py y reporte.py usan el mismo formato y el archivo se conserva
    entre ejecuciones (modo de agregar, un solo encabezado)."""
    ruta = tmp_path / "registros" / "registro_solicitudes.csv"
    registrar_solicitud(ruta, 200, "v1", 18.7)
    registrar_solicitud(ruta, 200, "v1", 19.2)
    registrar_solicitud(ruta, 422)

    lineas = ruta.read_text(encoding="utf-8").splitlines()

    assert lineas[0] == ENCABEZADO.strip()
    assert len(lineas) == 4
    assert calcular_metricas(ruta) == (3, pytest.approx(100 / 3))
