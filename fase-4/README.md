# Fase 4 — Monitoreo

Registra cada solicitud a `POST /predict` de la API de la fase 3 y calcula dos métricas de
operación: el **total de solicitudes** y el **porcentaje de solicitudes fallidas**. También
identifica qué versión del modelo predice, cómo volver a la anterior y cuándo reentrenar.

## Estructura

```text
fase-4/
├── registro.py              <- código de registro: agrega una fila por solicitud a /predict
├── reporte.py               <- script de monitoreo: total y % de fallos
├── registro_ejemplo.csv     <- muestra del registro de la ejecución documentada abajo
├── tests/
│   ├── conftest.py
│   └── test_reporte.py
└── README.md
```

## Registro de solicitudes

El middleware `registrar_predicciones` de `fase-3/api.py` envuelve la ruta `/predict` y sus
manejadores de error. Cuando ya conoce el resultado, llama a `registrar_solicitud()` de
`registro.py`. Así cada intento deja **una sola fila**, incluidos los rechazos por validación
(campo ausente, tipo inválido, JSON malformado) y los errores 503 o 500.

| Columna | Contenido |
| ------- | --------- |
| `fecha_hora` | Fecha y hora UTC (ISO 8601) |
| `estado_http` | Código devuelto |
| `version` | Versión que produjo la predicción; vacía si no se usó un modelo |
| `prediccion` | Minutos de retraso estimados; vacía en un error |

El archivo se abre en **modo de agregar** y el encabezado se escribe solo al crearlo, así que el
historial se conserva entre ejecuciones. La ubicación por defecto es
`fase-4/registro_solicitudes.csv` (no se versiona) y se cambia con `REQUEST_LOG_PATH`.

### Iniciar la API con el registro habilitado

El registro siempre está activo en `/predict`. Desde la raíz del repositorio, con el entorno
virtual activado:

```powershell
pip install -r fase-3/requirements.txt
cd fase-3
$env:MODEL_VERSION = "v1"
$env:REQUEST_LOG_PATH = "fase-4/registro_solicitudes.csv"   # ruta relativa a la raíz
uvicorn api:app --host 0.0.0.0 --port 8000
```

En otra terminal, desde `fase-3/`:

```powershell
python cliente_prueba.py
```

### Iniciar la API con Docker (registro persistente)

Con la imagen de la fase 3 (ver [`fase-3/README.md`](../fase-3/README.md#docker)), desde la raíz
del repositorio:

```bash
mkdir -p fase-4/registros
docker run -d --name api-retrasos -p 8000:8000 --user "$(id -u):$(id -g)" \
  -e MODEL_VERSION=v1 \
  -v "$PWD/fase-2/data:/app/fase-2/data:ro" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/fase-4/registros:/app/registros" \
  retrasos-vuelos:fase3
python3 fase-3/cliente_prueba.py
python3 fase-4/reporte.py --registro fase-4/registros/registro_solicitudes.csv
```

Dentro del contenedor la imagen fija `REQUEST_LOG_PATH=/app/registros/registro_solicitudes.csv`. Con
el montaje anterior, el registro queda en el anfitrión en
**`fase-4/registros/registro_solicitudes.csv`** (ignorado por Git) y sobrevive a `docker rm` y a
cambios de versión. La ejecución con Docker del 20 de noviembre de 2026 dio el mismo resultado que
el reporte de abajo (3 solicitudes, 33,3 % de fallos):
[`evidencia/fase-3/04_reporte_registro.txt`](../evidencia/fase-3/04_reporte_registro.txt). La
persistencia y el rollback a v2 están en
[`evidencia/fase-3/05_persistencia_rollback.txt`](../evidencia/fase-3/05_persistencia_rollback.txt).

### Calcular las métricas

Desde la raíz del repositorio:

```powershell
python fase-4/reporte.py --registro fase-4/registro_solicitudes.csv
```

`reporte.py` cuenta las filas de datos y las que tienen un código entre 400 y 599, divide la
segunda cifra por la primera y multiplica por 100. Si el registro está vacío (o solo tiene el
encabezado), muestra `Total de solicitudes: 0` y `sin solicitudes`.

## Reporte

**Ejecución del 19 de noviembre de 2026.** La API arrancó sobre un registro vacío con
`MODEL_VERSION=v1`, y `cliente_prueba.py` envió:

| # | Solicitud | Respuesta |
| - | --------- | --------- |
| 1 | Vuelo MQ DFW→RAP, salida 10:18, abril, jueves (fila 1 de `fase-2/data/prediccion.csv`) | 200 · `-1.33` min · `v1` |
| 2 | Vuelo AA CLT→PHX, salida 16:37, enero, lunes (fila 2) | 200 · `22.36` min · `v1` |
| 3 | El vuelo 1 **sin `distance`** | 422 · `Faltan variables predictoras obligatorias: ['distance']` |

Registro resultante (copiado en `registro_ejemplo.csv`):

```text
fecha_hora,estado_http,version,prediccion
2026-11-20T00:21:15+00:00,200,v1,-1.3319572571481464
2026-11-20T00:21:17+00:00,200,v1,22.363783673281336
2026-11-20T00:21:19+00:00,422,,
```

Salida del script de monitoreo:

```text
$ python fase-4/reporte.py --registro fase-4/registro_solicitudes.csv
Total de solicitudes: 3
Porcentaje de solicitudes fallidas: 33,3 %
```

| Indicador | Valor |
| --------- | ----: |
| Total de solicitudes | **3** |
| Solicitudes fallidas | **1 / 3 (33,3 %)** |

**Interpretación.** La API procesó dos vuelos con v1 y rechazó el tercero porque faltaba
`distance`, de modo que la tasa de fallos fue 1/3 (33,3 %). Ese rechazo era el resultado
esperado de la prueba de validación y no indica un fallo del servicio. El porcentaje
describe el funcionamiento de la API, no la calidad de las predicciones: para medirla se
necesitaría el `arr_delay` real de esos vuelos.

En la misma ejecución, `POST /train` creó la versión **v3** (201). Esa llamada no se registra
porque el registro cubre solo `/predict`.

## Versiones del modelo

| Versión | Estado | Origen | Partición | Semilla | Faltantes inducidos | MAE test (baseline) | RMSE test | Entorno | Commit |
| ------- | ------ | ------ | --------- | ------: | ------------------: | ------------------: | --------: | ------- | ------ |
| **v1** | **Activa** | fase 2, `train.py` | aleatoria | 42 | 1 % | 29.12 (24.56) | 50.85 | Python 3.13.7, scikit-learn 1.9.0 | `d934afb` |
| **v2** | **Anterior / rollback** | fase 2, `train.py` | por fecha | 42 | 1 % | 27.81 (23.85) | 52.28 | Python 3.13.7, scikit-learn 1.9.0 | `d934afb` |
| v3 | Candidata | fase 3, `POST /train` | aleatoria | 42 | 0 % | 29.08 (24.56) | 50.82 | Python 3.13.7, scikit-learn 1.9.0 | `b4b1f5a` |

Las tres versiones están en `fase-2/models/<versión>/`, con `modelo.joblib`, `metrics.json` y
`metadata.json`. Todas se entrenaron con el mismo CSV (`flight_data_2024_sample.csv`,
SHA-256 `3474b057…`).

**Qué versión usa la API.** La que indica `MODEL_VERSION` (por defecto `v1`). Se comprueba con
`GET /health` (`version_configurada`) y en cada respuesta y cada fila del registro (`version`).

**Cómo volver a la anterior.** Basta con reiniciar la API con la otra versión. No hay que
copiar ni borrar archivos:

```powershell
$env:MODEL_VERSION = "v2"   # o "v1" para volver
uvicorn api:app --host 0.0.0.0 --port 8000
```

Con Docker, se recrea el contenedor con la otra versión (los montajes no cambian):

```bash
docker rm -f api-retrasos
docker run -d --name api-retrasos -p 8000:8000 --user "$(id -u):$(id -g)" -e MODEL_VERSION=v2 \
  -v "$PWD/fase-2/data:/app/fase-2/data:ro" -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/fase-4/registros:/app/registros" retrasos-vuelos:fase3
curl -s http://localhost:8000/health   # "version_configurada":"v2"
```

> **Las métricas de la tabla no se comparan directamente.** v1 y v2 difieren en la partición:
> v2 evalúa días no vistos, que es una prueba más exigente, y su conjunto de prueba es
> distinto. v1 y v3 comparten partición y semilla, pero v3 se entrenó sin el 1 % de faltantes
> inducidos. Por eso la pequeña diferencia de MAE entre ellas refleja datos de entrenamiento
> distintos y no una mejora del modelo. En las tres, la regresión lineal sigue sin superar al
> baseline de la mediana en MAE (la conclusión de las fases 1 y 2).

## Política de reentrenamiento

Revisaremos el modelo cada vez que BTS publique un mes nuevo de datos On-Time con `arr_delay`
real (la frecuencia de publicación del origen), siempre que aporte al menos 1 000 vuelos
nuevos con llegada observada. El responsable de modelos y evaluación ejecutará `POST /train`
para crear una versión candidata, y evaluará la candidata y la versión activa con `predict.py`
sobre **los mismos vuelos reservados de ese mes**, que ninguna de las dos vio al entrenar,
usando el **MAE** como métrica principal de la fase 2. Solo se activará la candidata (cambiando
`MODEL_VERSION`) si reduce el MAE de la versión activa y además supera al baseline de la
mediana. En caso contrario se conserva la versión activa y la candidata queda guardada sin
usarse.

Los rechazos por validación, como el campo ausente del reporte, no son un motivo para
reentrenar: se corrigen en el cliente que envía los datos.

## Pruebas

Desde `fase-4/`:

```powershell
python -m pytest -v
```

Las pruebas comprueban el ejemplo de la especificación (3 solicitudes, 33,3 %), el registro
vacío (total 0, «sin solicitudes»), que 4xx y 5xx cuentan como fallos y que `reporte.py` lee el
formato que escribe `registro.py`. Que la API deje **una fila por intento** lo prueba
`test_registro_guarda_una_fila_por_intento` en `fase-3/tests/`.
