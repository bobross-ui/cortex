# Backend image for one-command local runs (`docker compose up`).
FROM python:3.11-slim

WORKDIR /app

# Install the package and its dependencies first so this layer caches across
# source edits. Only the files the install needs are copied at this stage.
COPY pyproject.toml ./
COPY backend ./backend
RUN pip install --no-cache-dir -e .

# Copy the rest (bundled fixture, etc.) so the image runs without a bind mount.
# Compose bind-mounts the repo over /app at runtime for live source + .env.
COPY . .

# BGE model cache lives here; compose mounts a volume so it downloads only once.
ENV HF_HOME=/hf

EXPOSE 8000

CMD ["uvicorn", "cortex.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
