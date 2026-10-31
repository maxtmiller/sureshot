FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY sureshot ./sureshot
RUN pip install --no-cache-dir ".[api]"
CMD ["uvicorn", "sureshot.api:app", "--host", "0.0.0.0", "--port", "8000"]
