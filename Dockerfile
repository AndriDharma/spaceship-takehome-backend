FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run supplies PORT. Defaulted so the image also runs locally.
ENV PORT=8080
EXPOSE 8080

# One worker. The application is I/O bound - database round trips and Vertex
# calls - and a second worker on the smallest Cloud Run instance would double
# the connection pool against the smallest Cloud SQL tier for no throughput.
#
# --timeout-keep-alive is raised because an SSE connection is idle between
# events and the default would close a slow turn mid-stream.
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 120
