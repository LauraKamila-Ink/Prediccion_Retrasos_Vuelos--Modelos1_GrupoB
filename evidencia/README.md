# Evidencia de ejecución — Docker (fases 2, 3 y 4)

Todos los archivos son la salida real de los comandos que se muestran en su primera línea
(precedidos por `$`), capturada con redirección a archivo. No se editaron a mano.

| Dato | Valor |
| ---- | ----- |
| Fecha | 20 de noviembre de 2026 (UTC−5) |
| Commit de la imagen y del clon | `f4570a4fb646a8c069fa93499d2691f02f44af07` (`build(fase-3): agregar Dockerfile de la API`) |
| Sistema operativo | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, x86_64 (máquina virtual) |
| Docker | Docker Engine Community 29.10.0 (cliente y servidor) |
| Python en las imágenes | 3.11.18 (`python:3.11-slim`), scikit-learn 1.9.0 |
| Datos | `flight_data_2024_sample.csv`, SHA-256 `3474b0573b40c922e97ae270b7ec6a9ab296ebd8d87767696d9cb7a030856264` |

## `fase-3/` — API en Docker (fases 3 y 4)

Desde la raíz del repositorio, con el registro `fase-4/registros/registro_solicitudes.csv` vacío al
empezar.

| Archivo | Qué demuestra |
| ------- | ------------- |
| [`00_entorno.txt`](fase-3/00_entorno.txt) | Fecha, commit, árbol limpio, SO, `docker version`, hash de los datos y SHA-256 de v1, v2 y v3 **antes** de la ejecución. |
| [`01_build.txt`](fase-3/01_build.txt) | `docker build -f fase-3/Dockerfile` desde la raíz; contexto de 46 kB (datos y modelos excluidos), `exit=0`. |
| [`02_run_health.txt`](fase-3/02_run_health.txt) | `docker run` con `-p 8000:8000`, `--user`, datos `:ro`, modelos con escritura y registro montado; `GET /health` → v1, `modelo_disponible: true`. |
| [`03_cliente_train.txt`](fase-3/03_cliente_train.txt) | `cliente_prueba.py --train`: 2 × 200 con v1, 1 × 422 (falta `distance`) y `POST /train` → 201 **v4**. `v4` aparece en el anfitrión a nombre del usuario y su `metadata.json` tiene `codigo.commit` = `GIT_COMMIT` de la imagen y `arbol_limpio: null`. |
| [`v4_metadata.json`](fase-3/v4_metadata.json), [`v4_metrics.json`](fase-3/v4_metrics.json) | Artefactos de texto de v4. El `.joblib` no se versiona y v4 se borró después: v3 sigue siendo la candidata documentada. |
| [`04_reporte_registro.txt`](fase-3/04_reporte_registro.txt), [`registro_solicitudes.csv`](fase-3/registro_solicitudes.csv) | `reporte.py` sobre el registro persistido: **3 solicitudes, 33,3 % de fallos**, que coincide con las 3 filas guardadas (200, 200, 422). |
| [`05_persistencia_rollback.txt`](fase-3/05_persistencia_rollback.txt) | (1) Tras `docker rm -f` y un contenedor nuevo, las 3 filas siguen en el CSV. (2) Rollback con `MODEL_VERSION=v2`: `/health` y `/predict` responden `v2`. (3) Con `MODEL_VERSION=v99`, `/health` da `modelo_disponible: false` y `/predict` → **503**. El registro acumula 5 filas (40,0 %), porque estas dos solicitudes extra se suman a las 3 del ensayo. |
| [`06_hashes_despues.txt`](fase-3/06_hashes_despues.txt) | SHA-256 de v1, v2 y v3 **idénticos** a los de `00_entorno.txt`: `/train` no tocó las versiones anteriores. |
| [`07_docker_logs.txt`](fase-3/07_docker_logs.txt) | Log de uvicorn del primer contenedor (arranque en `0.0.0.0:8000` y cada solicitud). |
| [`08_pytest.txt`](fase-3/08_pytest.txt) | `python -m pytest` dentro de la imagen con el repositorio montado en solo lectura: **15, 20 y 5** pruebas satisfactorias en `fase-2`, `fase-3` y `fase-4`. |

Las métricas de v4 son las mismas de v3 (misma configuración de `/train`: partición aleatoria,
semilla 42, sin faltantes inducidos). Solo cambia el último dígito de coma flotante del MAE de
entrenamiento, por la diferencia de Python (3.13.7 en v3, 3.11.18 en v4).

## `fase-2/` — Recorrido desde un directorio limpio (retroalimentación de la fase 2)

Lo pidió el profesor: entrenar con un identificador nuevo, cargar el artefacto, predecir sobre un CSV
pequeño, conservar salidas, métricas y metadatos, y comprobar que repetir el identificador no
reemplaza la versión. Se clonó la rama `feature/fase3-docker` en una carpeta nueva, se copió solo el
CSV de Kaggle y todo se ejecutó con la imagen de la fase 2 (`fase-2/Dockerfile`).

| Archivo | Qué demuestra |
| ------- | ------------- |
| [`00_clon.txt`](fase-2/00_clon.txt) | Clon nuevo con `git status` vacío, solo `prediccion.csv` en `data/` y v1 a v3 en `models/`; se copia el CSV y se comprueba su hash. |
| [`01_build.txt`](fase-2/01_build.txt) | `docker build -t retrasos-vuelos:fase2 .` desde `fase-2/`, `exit=0`. |
| [`02_train.txt`](fase-2/02_train.txt) | `train.py --version v-evidencia --particion fecha` en Docker: métricas impresas y los tres artefactos en `models/v-evidencia/` del anfitrión. |
| [`v-evidencia_metrics.json`](fase-2/v-evidencia_metrics.json), [`v-evidencia_metadata.json`](fase-2/v-evidencia_metadata.json) | Métricas y metadatos guardados (`codigo.commit` = `f4570a4…`, pasado con `-e GIT_COMMIT`). El `.joblib` no se versiona. |
| [`03_predict.txt`](fase-2/03_predict.txt), [`predicciones.csv`](fase-2/predicciones.csv) | `predict.py --version v-evidencia` carga el artefacto recién creado y predice las 10 filas de `prediccion.csv`. |
| [`04_duplicado_hashes.txt`](fase-2/04_duplicado_hashes.txt) | Repetir `train.py --version v-evidencia` falla con `Error: la versión 'v-evidencia' ya existe` (`exit=1`), y el SHA-256 del modelo es el mismo antes y después. |

**Cómo leer las métricas de `v-evidencia` frente a v2.** Ambas usan la partición por fecha con
semilla 42, así que comparten los mismos 74 días de prueba (baseline idéntico: MAE 23.85). La única
diferencia es que v2 se entrenó con un 1 % de faltantes inducidos en `crs_elapsed_time` y
`v-evidencia` sin ellos (el valor por defecto actual de `train.py`). Por eso las métricas son casi
iguales, pero no idénticas:

| Versión | Faltantes inducidos | MAE test (baseline) | RMSE test | Mejora MAE |
| ------- | ------------------: | ------------------: | --------: | ---------: |
| v2 | 1 % | 27.809 (23.852) | 52.276 | −16.59 % |
| v-evidencia | 0 % | 27.809 (23.852) | 52.269 | −16.59 % |

La conclusión de las fases 1 y 2 se mantiene: la regresión lineal no supera al baseline de la
mediana en MAE.

## Cómo reproducir

Los comandos exactos están en la primera línea de cada archivo y documentados en
[`fase-3/README.md`](../fase-3/README.md#docker) (API) y en el [README raíz](../README.md)
(fase 2 con Docker, en bash).
