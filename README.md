## Overview

This repo is a small FastAPI harness to compare **LangChain DeepAgents** vs the **Claude Agent SDK** while running **locally** against **LM Studio**.

It exposes two streaming endpoints:

- `POST /agents/deepagents-stream`: DeepAgents stream
- `POST /agents/claude-stream`: Claude Agent SDK stream (SSE)

The goal is to measure latency/throughput (e.g. TTFT, total time, tokens/sec) using the same local model backend.

## Prerequisites

- **Python**: 3.13+
- **uv** installed (`pipx install uv` or see uv docs)
- **LM Studio** running a local server (OpenAI-compatible API)

## Setup

1. Start LM Studio.
2. Go to **Local Server** and start the server (default `http://localhost:1234`).
3. Load your model (example used here: `qwen/qwen3.5-35b-a3b`).
4. Create a `.env` file (or copy `example.env`) and make sure it points at LM Studio. Example values for a fully local, no-real-keys setup:

   ```properties
   ANTHROPIC_BASE_URL=http://localhost:1234
   ANTHROPIC_AUTH_TOKEN=lmstudio

   OPENAI_API_KEY=lmstudio
   OPENAI_BASE_URL=http://localhost:1234/v1

   LM_STUDIO_MODEL=qwen/qwen3.5-35b-a3b
   ```

   These are **safe, local-only placeholders**:
   - `OPENAI_API_KEY` / `ANTHROPIC_AUTH_TOKEN` can be any non-empty string when talking to LM Studio; nothing is sent to a cloud API.
   - `.env` is already in `.gitignore` so it will not be committed by default.

Install deps:

```bash
uv sync
```

If you ever switch to real cloud APIs (OpenAI, Anthropic, etc.):
- Put real keys only in `.env` (never in code or `example.env`).
- Keep `.env` out of version control (already enforced by `.gitignore` in this repo).

## Run the API

```bash
uv run fastapi dev main.py --port 8001
```

Open docs at `http://127.0.0.1:8001/docs`.

### Quick curl test

DeepAgents stream:

```bash
curl -N -sS -X POST "http://127.0.0.1:8001/agents/deepagents-stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":"Say hi in one short sentence."}'

curl -N -sS -X POST "http://127.0.0.1:8001/agents/deepagents-stream" \
-H "Content-Type: application/json" \
-d '{"messages":"Say hi to me as if it is a poem about love."}'
```

Claude Agent SDK stream (SSE):

```bash
curl -N -sS -X POST "http://127.0.0.1:8001/agents/claude-stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":"Say hi in one short sentence."}'

curl -N -sS -X POST "http://127.0.0.1:8001/agents/claude-stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":"Say hi to me as if it is a poem about love."}'
```

## Run the benchmark

`benchmark_agents.py` benchmarks both FastAPI endpoints (and optionally a raw LM Studio baseline).

It reports:

- **TTFT** (time to first token)
- **Total time**
- **Tok/s** (best-effort)
- **p50 / p90 / p95** percentiles (warmups excluded)
- Optional **ΔTTFT / ΔTotal** vs the raw LM Studio baseline

Examples:

```bash
# Benchmark both endpoints (1 round), no raw LM Studio baseline
uv run python benchmark_agents.py --api-base-url http://127.0.0.1:8001 --rounds 1 --skip-raw

# Benchmark both endpoints (3 rounds) + raw LM Studio baseline
uv run python benchmark_agents.py --api-base-url http://127.0.0.1:8001 --rounds 3

# More stable numbers: 2 warmups + 10 measured rounds
uv run python benchmark_agents.py --api-base-url http://127.0.0.1:8001 --warmup 2 --rounds 12

# Basic load test (watch p95 + error rate)
uv run python benchmark_agents.py --api-base-url http://127.0.0.1:8001 --rounds 3 --warmup 1 --concurrency 5 --timeout-s 60
```

Useful flags:

- `--api-base-url`: where FastAPI is running (default `http://127.0.0.1:8001`)
- `--rounds`: how many times to run each prompt (default `3`)
- `--warmup`: warmup rounds (excluded from summary) (default `1`)
- `--concurrency`: concurrent requests (default `1`)
- `--timeout-s`: request timeout per run (default `180`)
- `--skip-raw`: skip the direct LM Studio baseline
- `--skip-deepagents`, `--skip-claude`: benchmark only one endpoint

Results are saved to `benchmark_results.json`.

### Benchmarking tips (to get clean metrics)

- Run the API **without auto-reload** during benchmarking. Auto-reload can interrupt streaming responses and skew results.
- Prefer reporting **p50/p90/p95** (not only averages) and use `--warmup` to exclude cold-start effects.

## Security notes

If you fork this repo:
- Keep `.env` out of Git.
- Do not put real API keys into `example.env`, the blog post, or screenshots.
- Be explicit in docs when enabling shell/web tools so users understand the implications.