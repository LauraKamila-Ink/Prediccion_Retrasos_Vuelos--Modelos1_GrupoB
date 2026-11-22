# Imagen de la API de la fase 3. Se construye desde la RAÍZ del repositorio:
#   docker build -f fase-3/Dockerfile --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
#     -t retrasos-vuelos:fase3 .
FROM python:3.11-slim

WORKDIR /app

# Se respeta la estructura del repositorio: api.py importa de ../fase-2 y ../fase-4,
# y fase-3/requirements.txt incluye -r ../fase-2/requirements.txt
COPY fase-2/requirements.txt fase-2/requirements.txt
COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

COPY fase-2/pipeline.py fase-2/predict.py fase-2/train.py fase-2/
COPY fase-4/registro.py fase-4/reporte.py fase-4/
COPY fase-3/api.py fase-3/cliente_prueba.py fase-3/

# Dentro de la imagen no hay .git: train.py guarda este commit en metadata.json
ARG GIT_COMMIT=""
# HOME=/tmp permite ejecutar con --user "$(id -u):$(id -g)" (UID sin carpeta personal)
ENV GIT_COMMIT=${GIT_COMMIT} \
    MODEL_VERSION=v1 \
    REQUEST_LOG_PATH=/app/registros/registro_solicitudes.csv \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Puntos de montaje: datos (solo lectura), modelos (escritura para /train) y registro
RUN mkdir -p /app/fase-2/data /app/fase-2/models /app/registros

WORKDIR /app/fase-3
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
