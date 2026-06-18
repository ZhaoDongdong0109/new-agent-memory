# Linux and Docker Quickstart

This project is pure Python, so the Linux version is the same code running in a
cleaner runtime environment. Docker is the easiest way to get repeatable Linux
behavior on Windows, macOS, or Linux.

## 1. Configure API

Copy the example file and fill in your endpoint:

```bash
cp .env.example .env
```

For Hermes, LM Studio, vLLM, Ollama gateway, or another OpenAI-compatible API:

```bash
OPENAI_API_KEY=your-key-or-local-placeholder
OPENAI_BASE_URL=http://host.docker.internal:1234/v1
OPENAI_MODEL=your-model-name
```

On native Linux, if the model server is on the same machine, use:

```bash
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
```

In Docker Desktop on Windows/macOS, `host.docker.internal` usually points from
the container back to the host machine.

## 2. Run With Docker Compose

Interactive Cognitive Runtime chat:

```bash
docker compose run --rm agent
```

One-shot runtime task:

```bash
docker compose run --rm agent ask --runtime --max-steps 4 \
  "Answer first, then call introspect, then finish with two next questions."
```

Single-step agent mode:

```bash
docker compose run --rm agent ask "Who are you?"
```

Memory is persisted in `./memory_data` on the host.

## 3. Run On Native Linux Or WSL

```bash
chmod +x scripts/linux_chat.sh
./scripts/linux_chat.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
new-agent-memory chat --runtime --max-steps 4
```

## 4. Useful Commands

Run tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Build image:

```bash
docker compose build
```

Open a shell in the Linux container:

```bash
docker compose run --rm --entrypoint bash agent
```
