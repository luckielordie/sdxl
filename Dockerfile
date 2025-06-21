FROM python:3.10-slim

# Set environment variables for Poetry to work smoothly in Docker
# - Don't create a virtual environment inside the container
# - Disable interactive prompts
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root

COPY . .

ENTRYPOINT ["poetry", "run", "python", "main.py"]
CMD [ "-h" ]