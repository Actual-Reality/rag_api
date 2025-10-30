FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS main

WORKDIR /app

# Install pandoc and netcat
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    pandoc \
    netcat-openbsd \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Download standard NLTK data, to prevent unstructured from downloading packages at runtime
RUN /app/.venv/bin/python -m nltk.downloader -d /app/nltk_data punkt_tab averaged_perceptron_tagger
ENV NLTK_DATA=/app/nltk_data

# Disable Unstructured analytics
ENV SCARF_NO_ANALYTICS=true

COPY . /app

CMD ["/app/.venv/bin/python", "main.py"]
