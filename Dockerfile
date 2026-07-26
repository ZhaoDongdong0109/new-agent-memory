FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY core ./core
COPY new_agent_memory ./new_agent_memory
COPY forgotten_layer.py main.py memory_chunk.py memory_layer_core.py retrieval.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p /app/memory_data \
    && chown -R agent:agent /app

USER agent

VOLUME ["/app/memory_data"]

ENTRYPOINT ["new-agent-memory"]
CMD ["chat", "--runtime", "--max-steps", "4"]
