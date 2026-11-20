# Fase 3 — API REST

API con FastAPI que **consulta** los modelos versionados de la fase 2 y **entrena** versiones
nuevas mediante solicitudes HTTP. No se desarrolla un modelo nuevo: la API reutiliza
`predecir()` y `validar_datos_entrada()` de `fase-2/predict.py`, `entrenar_version()` de
`fase-2/train.py` y el pipeline guardado en cada `modelo.joblib`, que incluye la ingeniería de
características, la imputación por mediana, el escalado y el One-Hot.

> **Por qué la entrada solo tiene variables pre-vuelo.** El contrato de `/predict` acepta
> únicamente las nueve predictoras que se conocen antes del despegue. Así la API sirve en el
> momento útil, antes de que el vuelo salga. Si el modelo necesitara `dep_delay` o
> `taxi_out`, solo podría consultarse cuando ya es tarde. La API rechaza explícitamente
> `arr_delay` y las columnas posteriores al vuelo.

## Estructura

```text
fase-3/
├── api.py              <- aplicación FastAPI (/health, /predict, /train)
├── cliente_prueba.py   <- cliente de prueba (solo biblioteca estándar)
├── requirements.txt    <- dependencias fijadas (incluye las de fase-2)
├── tests/
│   ├── conftest.py
│   └── test_api.py     <- pruebas con TestClient
└── README.md
```

La API importa código de `fase-2/` (pipeline, predicción y entrenamiento) y de `fase-4/`
(registro de solicitudes), por eso se ejecuta dentro del repositorio completo.

## Puesta en marcha

Desde la raíz del repositorio, con el entorno virtual activado (ver el README raíz):

```powershell
pip install -r fase-3/requirements.txt
cd fase-3
uvicorn api:app --host 0.0.0.0 --port 8000
```

La documentación interactiva queda en <http://localhost:8000/docs>.

### Configuración

| Variable | Por defecto | Uso |
| -------- | ----------- | --- |
| `MODEL_VERSION` | `v1` | Versión que atiende `/predict` (carpeta dentro de `MODELS_DIR`). |
| `MODELS_DIR` | `fase-2/models` | Carpeta de las versiones guardadas. `/train` escribe aquí. |
| `TRAIN_DATA_PATH` | `fase-2/data/flight_data_2024_sample.csv` | CSV que usa `/train` (se descarga de Kaggle, ver el README raíz). |
| `REQUEST_LOG_PATH` | `fase-4/registro_solicitudes.csv` | Registro de solicitudes a `/predict` (fase 4). |

Las rutas relativas se resuelven desde la raíz del repositorio. La configuración se lee en
cada solicitud.

### Seleccionar una versión

Las versiones disponibles son las carpetas de `fase-2/models/` que contienen `modelo.joblib`
(`v1`, `v2` y `v3`). Para usar otra, se reinicia la API con `MODEL_VERSION`:

```powershell
$env:MODEL_VERSION = "v2"     # PowerShell; en bash: MODEL_VERSION=v2 uvicorn ...
uvicorn api:app --host 0.0.0.0 --port 8000
```

`GET /health` informa cuál está configurada (`version_configurada`) y si existe
(`modelo_disponible`). Si la versión no existe, `/predict` responde **503** con un mensaje que
indica cómo corregirlo.

## Endpoints

| Método | Ruta | Respuesta |
| ------ | ---- | --------- |
| `GET` | `/health` | 200: `{"estado": "en ejecución", "version_configurada": ..., "modelo_disponible": ...}` |
| `POST` | `/predict` | 200: `{"prediccion": <minutos de retraso>, "version": ...}` |
| `POST` | `/train` | 201: `{"version": "v<N>", "directorio": ..., "metricas_test": {...}, "version_activa": ...}` |

### Validación de `/predict`

El cuerpo debe tener **exactamente** las nueve predictoras: un número como número y una
categoría como texto.

| Situación | Código | Ejemplo de mensaje |
| --------- | ------ | ------------------ |
| Falta una variable | 422 | `Faltan variables predictoras obligatorias: ['distance']` |
| Tipo incompatible (`"835"` en `distance`) | 422 | `La variable 'distance' debe ser numérica o null.` |
| Fuera de rango (`crs_dep_time=1490`, `month=13`, `day_of_week=0`, distancia ≤ 0) | 422 | `La variable 'month' debe ser un entero entre 1 y 12.` |
| Columna extra, el objetivo o una variable posterior al vuelo | 422 | `Columnas no permitidas en la solicitud: ['arr_delay']` |
| `null` en una categórica | 422 | `La variable 'origin' debe ser un texto no vacío.` |
| `null` en una numérica | 200 | Se aplica la **imputación por mediana guardada en la fase 2**. |
| Versión configurada inexistente | 503 | `El modelo configurado 'v9' no está disponible...` |

Los aeropuertos o aerolíneas que no aparecen en el entrenamiento no se rechazan: el
`OneHotEncoder(handle_unknown="ignore")` de la fase 2 los codifica con ceros. La predicción no
necesita la variable objetivo.

### `/train`

Acepta un cuerpo vacío y entrena con la configuración de la fase 2: datos de
`TRAIN_DATA_PATH`, objetivo `arr_delay`, partición aleatoria 80/20, semilla 42 y sin faltantes
inducidos. Calcula el siguiente identificador libre (`v<N+1>`) y guarda `modelo.joblib`,
`metrics.json` y `metadata.json` (con el commit del código) en `MODELS_DIR`. **No modifica
las versiones anteriores ni cambia la versión activa**: activar la nueva es una decisión
explícita (`MODEL_VERSION`), según la política de la fase 4.

## Ejemplo de solicitud y respuesta

Ejecución del 19 de noviembre de 2026 con `MODEL_VERSION=v1`. El vuelo es la primera fila de
`fase-2/data/prediccion.csv`:

```powershell
python cliente_prueba.py --train
```

```text
### Estado del servicio
GET http://localhost:8000/health
HTTP 200: {"estado": "en ejecución", "version_configurada": "v1", "modelo_disponible": true}

### Predicción válida 1
POST http://localhost:8000/predict
Solicitud: {"op_unique_carrier": "MQ", "origin": "DFW", "dest": "RAP", "crs_dep_time": 1018, "crs_arr_time": 1149, "crs_elapsed_time": 151.0, "distance": 835.0, "month": 4, "day_of_week": 4}
HTTP 200: {"prediccion": -1.3319572571481464, "version": "v1"}

### Predicción válida 2
POST http://localhost:8000/predict
Solicitud: {"op_unique_carrier": "AA", "origin": "CLT", "dest": "PHX", "crs_dep_time": 1637, "crs_arr_time": 1923, "crs_elapsed_time": 286.0, "distance": 1773.0, "month": 1, "day_of_week": 1}
HTTP 200: {"prediccion": 22.363783673281336, "version": "v1"}

### Predicción inválida (falta distance)
POST http://localhost:8000/predict
Solicitud: {"op_unique_carrier": "MQ", "origin": "DFW", "dest": "RAP", "crs_dep_time": 1018, "crs_arr_time": 1149, "crs_elapsed_time": 151.0, "month": 4, "day_of_week": 4}
HTTP 422: {"detail": "Faltan variables predictoras obligatorias: ['distance']"}

### Entrenamiento
POST http://localhost:8000/train
HTTP 201: {"version": "v3", "directorio": "fase-2/models/v3", "metricas_test": {"modelo": {"mae": 29.078879673101284, "rmse": 50.82305572891518, "r2": -0.03803114293766807}, "baseline": {"mae": 24.5614837398374, "rmse": 51.74572858535541, "r2": -0.07606337280398767}}, "version_activa": "v1"}
```

Una predicción negativa significa que se estima una llegada **adelantada** (en minutos).

Con `curl` (bash):

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"op_unique_carrier":"MQ","origin":"DFW","dest":"RAP","crs_dep_time":1018,"crs_arr_time":1149,"crs_elapsed_time":151.0,"distance":835.0,"month":4,"day_of_week":4}'
curl -s -X POST http://localhost:8000/train
```

## Comprobación manual de `/train`

La llamada anterior creó `fase-2/models/v3/`, que se versiona en el repositorio:

```text
fase-2/models/v3/
├── metadata.json   <- commit b4b1f5a, datos sha256 3474b057..., partición aleatoria, semilla 42
├── metrics.json    <- MAE test 29.08 (baseline 24.56)
└── modelo.joblib
```

Los SHA-256 de `v1/modelo.joblib` y `v2/modelo.joblib` eran los mismos antes y después del
entrenamiento, así que las versiones anteriores se conservan intactas. La prueba
`test_train_crea_version_nueva_sin_modificar_las_anteriores` lo verifica en cada ejecución.

> v3 no es directamente comparable con v1, aunque usen la misma partición y la misma semilla.
> v1 se entrenó con un 1 % de faltantes inducidos en `crs_elapsed_time` y v3 sin ellos
> (`fraccion_faltantes=0`, el valor por defecto actual de la fase 2). La diferencia de MAE
> (29.12 → 29.08) se explica por ese cambio en los datos de entrenamiento y no indica una mejora del
> modelo.

## Pruebas

Desde `fase-3/`:

```powershell
python -m pytest -v
```

Las tres pruebas que pide la especificación son `test_health_responde_200`,
`test_predict_valido_devuelve_prediccion_y_version` y
`test_predict_sin_campo_obligatorio_se_rechaza`. La suite agrega tipos y rangos inválidos,
`null` numérico imputado, equivalencia con `predict.py`, selección de versión, el 503, el
registro de la fase 4 (una fila por intento) y `/train`. Las pruebas usan el modelo pequeño
`fase-2/models/v1` (17 KB) y escriben el registro y los entrenamientos en carpetas temporales.
Resultado actual: **20 pruebas satisfactorias**.

## Docker

> **Pendiente.** El `Dockerfile` de la fase 3 y la evidencia de construcción y ejecución se
> agregan en la rama `feature/fase3-docker`.
