FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY sureshot ./sureshot
RUN pip install --no-cache-dir ".[api,claude]"
# Log decisions by default, so the feedback loop has data; mount a volume at /data to keep them.
ENV SURESHOT_DB=/data/sureshot.db
VOLUME /data
EXPOSE 8000
CMD ["sureshot", "serve", "--host", "0.0.0.0", "--port", "8000"]
