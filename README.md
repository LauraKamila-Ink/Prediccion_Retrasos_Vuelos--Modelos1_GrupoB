# Predicción de retrasos de vuelos

Proyecto integrador de **Modelos y Simulación de Sistemas I** (2026-II, Grupo B).

Se predice el retraso en la llegada de vuelos comerciales domésticos de EE. UU.
(`arr_delay`, en minutos) usando únicamente información disponible **antes del despegue**.
Es un problema de regresión supervisada.

---

## Requisitos

- **Python 3.11 o superior** (el entorno actual usa 3.13.7).
- Entorno virtual. No instalar paquetes de forma global.
- Cuenta de Kaggle para descargar los datos.

## Puesta en marcha

```bash
# 1. Crear y activar el entorno virtual (desde la raíz del repositorio)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Datos

Los CSV **no se versionan** por su tamaño (están excluidos en `.gitignore`), así que hay
que descargarlos antes de ejecutar nada.

1. Descargar el dataset desde
   [Kaggle — Flight Delay Dataset 2024](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024).
2. Colocar los archivos en `fase-1/data/`, conservando los nombres originales:

```
fase-1/data/
├── flight_data_2024_sample.csv   ← muestra de 10 000 filas (la que usa el notebook)
└── flight_data_2024.csv          ← dataset completo, ~7 M filas (opcional, 1.3 GB)
```

El notebook solo necesita la muestra. El archivo completo se usa únicamente para
comprobaciones puntuales sobre la población.

## Ejecutar el notebook

```bash
cd fase-1
jupyter lab notebook.ipynb
```

Dos detalles que evitan errores:

- **Ejecutar siempre desde `fase-1/`.** El notebook lee los datos con la ruta relativa
  `data/flight_data_2024_sample.csv` y falla si el directorio de trabajo es otro.
- **Seleccionar el kernel del entorno virtual**, no el Python del sistema.

El notebook está preparado para correr de principio a fin con *Restart Kernel & Run All*,
sin intervención manual. El repositorio guarda el notebook **sin las salidas** (se limpian
automáticamente al hacer commit), así que hay que ejecutarlo para ver tablas y gráficos.

## Validar el dataset

Comprueba que los datos cumplen los requisitos de la especificación (número de
observaciones, predictoras, valores faltantes, ausencia de columnas con fuga):

```bash
cd fase-1
python validar_dataset.py --data data/flight_data_2024_sample.csv
```

Sobre el archivo completo conviene limitar las filas para no agotar la memoria:

```bash
python validar_dataset.py --data data/flight_data_2024.csv --nrows 500000
```

## Estructura

```
├── fase-1/          Modelo predictivo        (en curso)
├── fase-2/          Scripts y Docker
├── fase-3/          API REST
└── fase-4/          Monitoreo
```

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Modelo predictivo | Secciones 1–3 del notebook completas; preprocesamiento y modelado pendientes |
| 2 | Scripts y Docker | Pendiente |
| 3 | API REST | Pendiente |
| 4 | Monitoreo | Pendiente |
