FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY static/ static/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "scholar_mind.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
