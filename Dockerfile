FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOME=/home/app

WORKDIR /app

COPY requirements-runtime.txt ./requirements-runtime.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt

RUN addgroup --system appgroup \
    && adduser \
        --system \
        --ingroup appgroup \
        --home /home/app \
        appuser

COPY --chown=appuser:appgroup src ./src

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]