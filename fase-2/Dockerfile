FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py predict.py train.py ./

RUN mkdir -p /app/data /app/models /app/outputs

ENTRYPOINT ["python"]
CMD ["predict.py"]
