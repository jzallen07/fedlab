# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

CMD ["python", "-m", "src.server.app"]
