# Predicción de retrasos de vuelos

Proyecto integrador de **Modelos y Simulación de Sistemas I** (2026-II, Grupo B) — Universidad de Antioquia.

Se predice el retraso en la llegada de vuelos comerciales domésticos de EE. UU. (`arr_delay`, en minutos) utilizando únicamente información disponible **antes del despegue**.

El problema se aborda como un problema de **regresión supervisada**.

---

## Integrantes

| Nombre | GitHub |
| ------ | ------ |
| Laura Camila Fernandez Ospina | [`LauraKamila-Ink`](https://github.com/LauraKamila-Ink) |
| Mateo Vargas Tirado | [`AraiGumaDev`](https://github.com/AraiGumaDev) |

---

## Estado del proyecto

| Fase | Contenido         | Estado |
| ---- | ----------------- | ------ |
| 1    | Modelo predictivo | Completada (entrega 1) |
| 2    | Scripts y Docker  | En progreso: Parte 1 completada, Parte 2 pendiente |
| 3    | API REST          | Pendiente |
| 4    | Monitoreo         | Pendiente |

**Fase 2 — avance:**

* [x] Carpeta `fase-2/`, `requirements.txt` y reglas de `.gitignore`.
* [x] Pipeline de la fase 1 extraído a un módulo importable (`pipeline.py`).
* [x] `train.py` con parámetros por consola (datos, variable objetivo, versión, partición).
* [x] Modelos versionados en `models/v1` y `models/v2` con `modelo.joblib`, `metrics.json` y `metadata.json`.
* [ ] Archivo de prueba `prediccion.csv` y `predict.py` (salida en `outputs/predicciones.csv`).
* [ ] Pruebas automatizadas con `pytest` en `tests/`.
* [ ] `Dockerfile` y `README.md` de la fase 2.

---

## Objetivo y alcance

El objetivo es estimar el retraso de llegada de un vuelo a partir de variables conocidas antes de su salida.

Las variables predictoras utilizadas son:

* Aerolínea operadora (`op_unique_carrier`).
* Aeropuerto de origen (`origin`).
* Aeropuerto de destino (`dest`).
* Hora programada de salida (`crs_dep_time`).
* Hora programada de llegada (`crs_arr_time`).
* Duración programada del vuelo (`crs_elapsed_time`).
* Distancia (`distance`).
* Mes (`month`).
* Día de la semana (`day_of_week`).

La variable objetivo es `arr_delay`, expresada en minutos.

Para evitar **fuga de información (data leakage)**, no se utilizan variables que solamente pueden conocerse durante o después del vuelo, como el retraso real de salida, tiempos reales de vuelo o causas posteriores del retraso.

Los resultados aplican únicamente a **vuelos con llegada observada**: los vuelos cancelados o desviados no tienen `arr_delay` y se excluyen.

---

## Requisitos

* **Python 3.11 o superior**.
* El entorno utilizado durante el desarrollo es **Python 3.13.7**.
* Entorno virtual. No instalar paquetes de forma global.
* Cuenta de Kaggle para descargar los datos.

---

## Puesta en marcha

```bash
# 1. Crear y activar el entorno virtual (desde la raíz del repositorio)

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt           # fase 1: notebook (incluye Jupyter)
pip install -r fase-2/requirements.txt    # fase 2: solo dependencias de los scripts
```

---

## Datos

Los archivos CSV **no se versionan** debido a su tamaño y están excluidos mediante `.gitignore`. Por lo tanto, deben descargarse antes de ejecutar el proyecto.

### 1. Descargar los datos

El dataset utilizado se obtiene desde:

[Kaggle — Flight Delay Dataset 2024](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)

### 2. Ubicar los archivos

Los archivos deben conservar sus nombres originales y ubicarse en:

```text
fase-1/data/
├── flight_data_2024_sample.csv   <- muestra de 10 000 filas
└── flight_data_2024.csv          <- dataset completo, ~7 M filas (opcional)

fase-2/data/
└── flight_data_2024_sample.csv   <- copia de la misma muestra, para train.py
```

El notebook y `train.py` utilizan la muestra de **10 000 observaciones**.

El archivo completo se utiliza únicamente para comprobaciones puntuales sobre la población y no es necesario para ejecutar el flujo principal.

---

## Fase 1 — Modelo predictivo

### Metodología

1. **Exploración de los datos:** dimensiones, valores faltantes, distribución de `arr_delay`, valores extremos y análisis de variables categóricas y temporales.
2. **Preparación de los datos:** eliminación de vuelos sin variable objetivo, transformación cíclica de horas, meses y días de la semana, imputación por mediana, estandarización y codificación One-Hot.
3. **Separación train/test:** 80 % / 20 % aleatoria con `random_state = 42`.
4. **Prevención de fuga de información:** todas las transformaciones se ajustan únicamente con entrenamiento, dentro de un `Pipeline` con `ColumnTransformer`.
5. **Modelo base:** `DummyRegressor` que predice la **mediana** del retraso de entrenamiento.
6. **Modelo predictivo:** Regresión Lineal Múltiple sobre el pipeline anterior.
7. **Evaluación:** MAE (métrica principal), RMSE y R² en entrenamiento y prueba.
8. **Serialización:** el pipeline completo se guarda con `joblib`.

> **Valores faltantes inducidos.** Ninguna de las nueve predictoras tiene valores faltantes en el dataset original (0.00 % sobre 7 079 081 filas). Para aplicar una estrategia de imputación se indujo, de forma declarada y reproducible, un 1 % de `NaN` en `crs_elapsed_time` (sección 2.7 del notebook). No es una propiedad del dataset original.

### Resultados (conjunto de prueba)

| Modelo | MAE (min) | RMSE (min) | R² |
| ------ | --------: | ---------: | -: |
| Baseline (mediana) | 24.56 | 51.75 | −0.0761 |
| Regresión lineal | 29.12 | 50.85 | −0.0393 |

### Conclusión

La regresión lineal **no supera al baseline según el MAE**, la métrica principal: lo empeora un **18.55 %**, aunque reduce el RMSE un 1.72 %. La diferencia se explica por lo que optimiza cada modelo: `LinearRegression` minimiza el error cuadrático, por lo que se ajusta para reducir los errores grandes (RMSE), mientras que la mediana es el predictor constante que minimiza el error absoluto (MAE). Con las variables disponibles antes del despegue, la mediana sigue siendo la referencia preferible según el MAE.

### Ejecutar el notebook

```bash
cd fase-1

jupyter lab notebook.ipynb
```

* **Ejecutar siempre desde `fase-1/`.** El notebook utiliza la ruta relativa `data/flight_data_2024_sample.csv`.
* **Seleccionar el kernel del entorno virtual**, no el Python del sistema.
* El notebook está preparado para ejecutarse de principio a fin mediante **Restart Kernel & Run All**.
* El repositorio conserva el notebook **con sus salidas** (tablas, métricas y gráficos), por lo que puede consultarse sin ejecutarlo.

### Validar el dataset

Para comprobar que los datos cumplen los requisitos de la especificación (número de observaciones, variables predictoras, valores faltantes y ausencia de columnas con fuga de información):

```bash
cd fase-1

python validar_dataset.py --data data/flight_data_2024_sample.csv
```

Para validar una parte del dataset completo se recomienda limitar el número de filas:

```bash
python validar_dataset.py --data data/flight_data_2024.csv --nrows 500000
```

---

## Fase 2 — Scripts y Docker (Parte 1)

La fase 2 convierte el notebook en una aplicación que se entrena desde la línea de comandos. **No se desarrolla un modelo nuevo**: se reutiliza exactamente el pipeline de la fase 1.

### Estructura

```text
fase-2/
├── pipeline.py          <- módulo compartido: predictoras, ingeniería de características y pipeline
├── train.py             <- entrena, evalúa y guarda una versión del modelo
├── requirements.txt     <- dependencias de ejecución con versiones fijadas
├── .gitignore           <- excepciones de la fase (modelos versionados, outputs/)
├── data/                <- datos de entrada (CSV no versionados)
└── models/
    ├── v1/              <- partición aleatoria (reproduce la fase 1)
    │   ├── modelo.joblib
    │   ├── metrics.json
    │   └── metadata.json
    └── v2/              <- partición por fecha
        ├── modelo.joblib
        ├── metrics.json
        └── metadata.json
```

### `pipeline.py`

Contiene las constantes del problema (`PREDICTORAS`, `OBJETIVO_POR_DEFECTO`, `COLUMNAS_PROHIBIDAS`), la clase `IngenieriaCaracteristicas` (codificación cíclica de horas, mes y día de la semana) y las funciones `construir_modelo()` y `validar_columnas()`.

En la fase 1 la clase `IngenieriaCaracteristicas` estaba definida dentro del notebook, por lo que el modelo guardado **no podía cargarse fuera de esa sesión**. Al trasladarla a un módulo importable, el `modelo.joblib` se carga y predice desde cualquier script que se ejecute en `fase-2/`:

```python
import joblib
import pandas as pd

modelo = joblib.load("models/v1/modelo.joblib")
modelo.predict(pd.read_csv("data/flight_data_2024_sample.csv", nrows=5))
```

Además, el módulo falla al importarse si alguna predictora pertenece a `COLUMNAS_PROHIBIDAS`, lo que impide entrenar un modelo con fuga de información.

### `train.py`

Se ejecuta desde `fase-2/`:

```bash
cd fase-2

python train.py --data data/flight_data_2024_sample.csv --version v1 --particion aleatoria
python train.py --data data/flight_data_2024_sample.csv --version v2 --particion fecha
```

| Argumento | Por defecto | Descripción |
| --------- | ----------- | ----------- |
| `--data` | (obligatorio) | Ruta al CSV de vuelos. |
| `--version` | (obligatorio) | Nombre de la versión (`v1`, `v2`, …). Los artefactos se guardan en `models/<version>/`. |
| `--objetivo` | `arr_delay` | Variable objetivo. Debe ser numérica y no puede ser una predictora ni una columna posterior al vuelo. |
| `--particion` | `aleatoria` | `aleatoria`: `train_test_split` por filas. `fecha`: `GroupShuffleSplit` por `fl_date`. |
| `--test-size` | `0.2` | Fracción de prueba. Con `--particion fecha` es la fracción de **días**, no de filas. |
| `--random-state` | `42` | Semilla para el muestreo y la partición. |
| `--fraccion-faltantes` | `0.01` | Fracción de `NaN` inducidos en `crs_elapsed_time`, igual que en la fase 1. `0` los desactiva. |
| `--modelos-dir` | `models` | Carpeta raíz de los modelos versionados. |
| `--sobrescribir` | — | Permite reemplazar una versión existente. Sin esta opción, una versión ya guardada no se modifica. |

**Flujo del script:**

1. Lee el CSV y valida que estén las nueve predictoras y la variable objetivo (y `fl_date` si la partición es por fecha). Si falta alguna, termina con un mensaje que indica cuáles.
2. Induce los valores faltantes declarados, con la misma semilla y sobre las mismas filas que en la fase 1.
3. Descarta los vuelos sin objetivo (cancelados o desviados); el objetivo no se imputa.
4. Separa entrenamiento y prueba según `--particion`.
5. Entrena el baseline (mediana) y el pipeline de la fase 1, ambos solo con entrenamiento.
6. Imprime MAE, RMSE y R² de entrenamiento y prueba, la mejora frente al baseline y cuántos días de prueba también aparecen en entrenamiento.
7. Guarda los tres artefactos en `models/<version>/`.

### Artefactos de cada versión

* **`modelo.joblib`:** el `Pipeline` completo entrenado (ingeniería → imputación y escalado / One-Hot → regresión). Recibe un `DataFrame` con las nueve predictoras sin transformar y devuelve el retraso estimado en minutos.
* **`metrics.json`:** MAE, RMSE y R² de entrenamiento y prueba del modelo y del baseline, y la mejora porcentual del modelo frente al baseline en prueba (`mejora_mae_pct`, `mejora_rmse_pct`; un valor negativo significa que empeora).
* **`metadata.json`:** versión, fecha de entrenamiento, algoritmo, objetivo, predictoras, tipo de partición con su justificación, `test_size`, `random_state`, número de filas y de días en cada conjunto, nombre y SHA-256 del CSV utilizado, fracción de faltantes inducidos y versiones de Python, scikit-learn, pandas, numpy y joblib.

### Resultados v1 vs v2 (conjunto de prueba)

| Versión | Partición | Modelo MAE / RMSE / R² | Baseline MAE / RMSE | Mejora MAE | Días de prueba también en entrenamiento |
| ------- | --------- | ---------------------- | ------------------- | ---------: | --------------------------------------: |
| v1 | aleatoria | 29.12 / 50.85 / −0.0393 | 24.56 / 51.75 | −18.55 % | 362 de 362 |
| v2 | fecha | 27.81 / 52.28 / −0.0589 | 23.85 / 52.52 | −16.59 % | 0 de 74 |

**v1** reproduce exactamente las métricas de la fase 1. **v2** usa el mismo modelo, pero evalúa con días que el modelo no vio durante el entrenamiento.

### Justificación de la partición por fecha

Los retrasos se propagan dentro de un mismo día: el clima, la congestión de los aeropuertos y las aeronaves que encadenan varios vuelos afectan a muchos vuelos de la misma fecha. Con la partición aleatoria, los 362 días presentes en prueba también tienen vuelos en entrenamiento, así que las métricas de v1 describen vuelos **de días ya representados**.

La partición por fecha (`GroupShuffleSplit` agrupando por `fl_date`) garantiza que ningún día de prueba aparezca en entrenamiento y mide la generalización **a días no vistos del mismo periodo**. No es una evaluación hacia el futuro: los días de prueba están repartidos a lo largo de 2024.

Con ambas particiones la conclusión se mantiene: la regresión lineal no supera al baseline en MAE.

---

## Estructura del proyecto

```text
├── fase-1/          Modelo predictivo (notebook y validación del dataset)
├── fase-2/          Scripts y Docker
├── fase-3/          API REST
└── fase-4/          Monitoreo
```
