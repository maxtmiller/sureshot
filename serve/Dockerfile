FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY hunch ./hunch
RUN pip install --no-cache-dir ".[api]"
CMD ["uvicorn", "hunch.api:app", "--host", "0.0.0.0", "--port", "8000"]
