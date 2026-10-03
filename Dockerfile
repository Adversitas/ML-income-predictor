FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"
COPY configs ./configs
COPY tests ./tests

ENV PYTHONUNBUFFERED=1 \
    MLWF_DATA_DIR=/app/data \
    GIT_PYTHON_REFRESH=quiet
CMD ["mlwf", "--help"]
