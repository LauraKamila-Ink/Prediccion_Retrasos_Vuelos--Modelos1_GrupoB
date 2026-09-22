# Predicción de retrasos de vuelos

Proyecto integrador de **Modelos y Simulación de Sistemas I** (2026-II, Grupo B).

Se predice el retraso en la llegada de vuelos comerciales domésticos de EE. UU. (`arr_delay`, en minutos) utilizando únicamente información disponible **antes del despegue**.

El problema se aborda como un problema de **regresión supervisada**.

---

## Objetivo y alcance

El objetivo de la Fase 1 es construir y evaluar un modelo capaz de estimar el retraso de llegada de un vuelo a partir de variables conocidas antes de su salida.

Las variables predictoras utilizadas incluyen:

* Aerolínea operadora.
* Aeropuerto de origen.
* Aeropuerto de destino.
* Hora programada de salida.
* Hora programada de llegada.
* Duración programada del vuelo.
* Distancia.
* Mes.
* Día de la semana.

La variable objetivo es `arr_delay`, expresada en minutos.

Para evitar **fuga de información (data leakage)**, no se utilizan variables que solamente pueden conocerse durante o después del vuelo, como el retraso real de salida, tiempos reales de vuelo o causas posteriores del retraso.

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
pip install -r requirements.txt
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
```

El notebook utiliza principalmente la muestra de **10 000 observaciones**.

El archivo completo se utiliza únicamente para comprobaciones puntuales sobre la población y no es necesario para ejecutar el flujo principal de la Fase 1.

---

## Metodología de la Fase 1

El flujo de trabajo sigue las siguientes etapas:

### 1. Exploración de los datos

* Dimensiones y estructura del dataset.
* Análisis de valores faltantes.
* Distribución de `arr_delay`.
* Identificación de valores extremos.
* Análisis de variables categóricas y temporales.

### 2. Preparación de los datos

* Eliminación de vuelos sin variable objetivo disponible.
* Transformación cíclica de horas, meses y días de la semana.
* Imputación de valores faltantes en variables numéricas.
* Estandarización de variables numéricas.
* Codificación One-Hot de variables categóricas.

### 3. Separación Train/Test

* 80 % para entrenamiento.
* 20 % para prueba.
* `random_state = 42`.

### 4. Prevención de fuga de información

* Las transformaciones se ajustan únicamente con los datos de entrenamiento.
* El preprocesamiento y el modelo se integran mediante `Pipeline` y `ColumnTransformer`.
* Las variables que contienen información posterior al vuelo se excluyen del conjunto de predictores.

### 5. Modelo base

Se utiliza un `DummyRegressor` como referencia, utilizando la **mediana** del retraso de los vuelos del conjunto de entrenamiento como predicción constante.

### 6. Modelo predictivo

Se implementa una **Regresión Lineal Múltiple** con:

* Ingeniería de características temporales.
* Imputación de valores faltantes.
* Estandarización de variables numéricas.
* Codificación One-Hot de variables categóricas.
* Pipeline completo para garantizar un flujo reproducible.

### 7. Evaluación

Los modelos se evalúan utilizando:

* **MAE (Mean Absolute Error):** error absoluto medio en minutos.
* **RMSE (Root Mean Squared Error):** penaliza con mayor peso los errores grandes.
* **R² (Coeficiente de Determinación):** mide la proporción de variabilidad explicada por el modelo.

Se comparan las métricas del modelo base y del modelo predictivo tanto en entrenamiento como en prueba.

### 8. Serialización y verificación

El pipeline completo se serializa mediante `joblib`.

Posteriormente, el modelo se carga nuevamente y se verifica que produzca las mismas predicciones que el modelo original.

---

## Ejecutar el notebook

```bash
cd fase-1

jupyter lab notebook.ipynb
```

### Consideraciones importantes

* **Ejecutar siempre desde `fase-1/`.**

  El notebook utiliza la ruta relativa:

  ```text
  data/flight_data_2024_sample.csv
  ```

  por lo que puede producir errores si se ejecuta desde otro directorio.

* **Seleccionar el kernel del entorno virtual**, no el Python del sistema.

El notebook está preparado para ejecutarse de principio a fin mediante:

**Restart Kernel & Run All**

El repositorio conserva el notebook **sin las salidas**, por lo que es necesario ejecutarlo para visualizar nuevamente las tablas, métricas y gráficos generados.

---

## Validar el dataset

Para comprobar que los datos cumplen los requisitos de la especificación, como número de observaciones, variables predictoras, valores faltantes y ausencia de columnas con fuga de información, se puede ejecutar:

```bash
cd fase-1

python validar_dataset.py --data data/flight_data_2024_sample.csv
```

Para validar una parte del dataset completo se recomienda limitar el número de filas para reducir el consumo de memoria:

```bash
python validar_dataset.py --data data/flight_data_2024.csv --nrows 500000
```

---

## Estructura del proyecto

```text
├── fase-1/          Modelo predictivo
├── fase-2/          Scripts y Docker
├── fase-3/          API REST
└── fase-4/          Monitoreo
```

---

## Estado del proyecto

| Fase | Contenido         | Estado                                                               |
| ---- | ----------------- | -------------------------------------------------------------------- |
| 1    | Modelo predictivo | Completada                                                           |
| 2    | Scripts y Docker  | Pendiente                                                            |
| 3    | API REST          | Pendiente                                                            |
| 4    | Monitoreo         | Pendiente                                                            |

```

