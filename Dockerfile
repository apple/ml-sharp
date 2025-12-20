FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal build dependencies for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt requirements.in pyproject.toml /app/

RUN pip install --upgrade pip setuptools wheel && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy project
COPY . /app

# Default command prints the CLI help. Users can override with other commands.
CMD ["python", "-m", "sharp.cli.predict", "--help"]
