"""
benchmark_agents.py
===================
Benchmark: LangChain Deep Agents vs Claude Agent SDK
Focus: Tokens/sec · Time to first token · Total latency
Using your FastAPI app as the harness (both agents run locally via LM Studio)

Requirements
------------
  uv add httpx

LM Studio
---------
  1. Open LM Studio → Local Server → Start Server (default: http://localhost:1234)
  2. Load any GGUF model (e.g. qwen2.5-coder-7b-instruct)

Usage
-----
  # Run both frameworks, 3 rounds each
  python benchmark_agents.py --api-base-url http://127.0.0.1:8001

  # Change rounds/prompts
  python benchmark_agents.py --rounds 5
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx
import re


# ──────────────────────────────────────────────
# Shared data model
# ──────────────────────────────────────────────

@dataclass
class BenchResult:
    framework: str
    model: str
    prompt: str
    prompt_id: int
    run_id: int
    time_to_first_token_s: Optional[float]
    total_time_s: float
    total_tokens: Optional[int]
    tokens_per_second: Optional[float]
    is_warmup: bool = False
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Utility: pretty table
# ──────────────────────────────────────────────

def print_table(results: list[BenchResult]) -> None:
    cols = ["Framework", "Model", "TTFT (s)", "Total (s)", "Tok/s", "Tokens", "Error"]
    rows = []
    for r in results:
        rows.append([
            r.framework,
            r.model[:30],
            f"{r.time_to_first_token_s:.3f}" if r.time_to_first_token_s else "n/a",
            f"{r.total_time_s:.3f}",
            f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "n/a",
            str(r.total_tokens) if r.total_tokens else "n/a",
            r.error or "",
        ])
    widths = [max(len(str(row[i])) for row in [cols] + rows) for i in range(len(cols))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt_row(row):
        return "|" + "|".join(f" {str(v).ljust(w)} " for v, w in zip(row, widths)) + "|"
    print(sep)
    print(fmt_row(cols))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)

def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

def _fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


# ──────────────────────────────────────────────
# 1. DeepAgents via FastAPI endpoint
# ──────────────────────────────────────────────

async def bench_deepagents_endpoint(
    prompt: str,
    prompt_id: int,
    run_id: int,
    is_warmup: bool,
    api_base_url: str,
    timeout_s: float,
) -> BenchResult:
    """
    Streams from POST /agents/deepagents-stream (newline-delimited JSON chunks).
    """
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            async with client.stream(
                "POST",
                f"{api_base_url}/agents/deepagents-stream",
                json={"messages": prompt},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    # Common shape from deepagents stream in this repo:
                    # {"model": {"messages": ["content=\"...\" ... token_usage ..."]}}
                    model_obj = chunk.get("model")
                    if isinstance(model_obj, dict):
                        msgs = model_obj.get("messages")
                        if isinstance(msgs, list):
                            for msg in msgs:
                                if not isinstance(msg, str):
                                    continue
                                # Prefer model-reported completion tokens if present.
                                m = re.search(r"'completion_tokens':\s*(\d+)", msg)
                                if m:
                                    token_count += int(m.group(1))
                                    continue
                                # Fallback: try to extract the content="...".
                                m = re.search(r'content="(.*?)"', msg, flags=re.DOTALL)
                                if m:
                                    token_count += len(m.group(1).split())
                                else:
                                    token_count += len(msg.split())
    except Exception as e:
        return BenchResult(
            framework="DeepAgents",
            model="(via FastAPI)",
            prompt=prompt,
            prompt_id=prompt_id,
            run_id=run_id,
            is_warmup=is_warmup,
            time_to_first_token_s=None,
            total_time_s=time.perf_counter() - t_start,
            total_tokens=None,
            tokens_per_second=None,
            error=str(e),
        )

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else None
    tps = token_count / total if total > 0 and token_count else None

    return BenchResult(
        framework="DeepAgents",
        model="(via FastAPI)",
        prompt=prompt,
        prompt_id=prompt_id,
        run_id=run_id,
        is_warmup=is_warmup,
        time_to_first_token_s=ttft,
        total_time_s=total,
        total_tokens=token_count or None,
        tokens_per_second=tps,
    )


# ──────────────────────────────────────────────
# 2. Claude Agent SDK via FastAPI endpoint
# ──────────────────────────────────────────────

async def bench_claude_sdk_endpoint(
    prompt: str,
    prompt_id: int,
    run_id: int,
    is_warmup: bool,
    api_base_url: str,
    timeout_s: float,
) -> BenchResult:
    """
    Streams from POST /agents/claude-stream (SSE: lines like 'data: {...}').
    """
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            async with client.stream(
                "POST",
                f"{api_base_url}/agents/claude-stream",
                json={"messages": prompt},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "assistant_text":
                        token_count += len(str(event.get("text", "")).split())
                    elif event.get("type") == "result":
                        break
    except Exception as e:
        return BenchResult(
            framework="ClaudeAgentSDK",
            model="(via FastAPI)",
            prompt=prompt,
            prompt_id=prompt_id,
            run_id=run_id,
            is_warmup=is_warmup,
            time_to_first_token_s=None,
            total_time_s=time.perf_counter() - t_start,
            total_tokens=None,
            tokens_per_second=None,
            error=str(e),
        )

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else None
    tps = token_count / total if total > 0 and token_count else None

    return BenchResult(
        framework="ClaudeAgentSDK",
        model="(via FastAPI)",
        prompt=prompt,
        prompt_id=prompt_id,
        run_id=run_id,
        is_warmup=is_warmup,
        time_to_first_token_s=ttft,
        total_time_s=total,
        total_tokens=token_count or None,
        tokens_per_second=tps,
    )


# ──────────────────────────────────────────────
# 3. Raw OpenAI-compat streaming baseline
#    (Direct LM Studio call, no framework overhead)
# ──────────────────────────────────────────────

async def bench_raw_lmstudio(
    prompt: str,
    prompt_id: int,
    run_id: int,
    is_warmup: bool,
    lm_studio_url: str,
    model_name: str,
    timeout_s: float,
) -> BenchResult:
    """
    Direct streaming POST to LM Studio – used as a baseline to measure
    how much overhead each framework adds.
    """
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    token_count = 0
    error_msg: Optional[str] = None

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            async with client.stream(
                "POST",
                f"{lm_studio_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer lm-studio"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if t_first is None:
                                t_first = time.perf_counter()
                            token_count += len(delta.split())
                    except (json.JSONDecodeError, KeyError):
                        pass
    except Exception as e:
        error_msg = str(e)

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else None
    tps = token_count / total if total > 0 and token_count else None

    return BenchResult(
        framework="RawLMStudio",
        model=model_name,
        prompt=prompt,
        prompt_id=prompt_id,
        run_id=run_id,
        is_warmup=is_warmup,
        time_to_first_token_s=ttft,
        total_time_s=total,
        total_tokens=token_count or None,
        tokens_per_second=tps,
        error=error_msg,
    )


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

PROMPTS = [
    "In two sentences, explain what a large language model is.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What are three key differences between REST and GraphQL APIs?",
]


async def run_all(args) -> list[BenchResult]:
    results: list[BenchResult] = []

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))

    async def _run_one(framework: str, prompt: str, prompt_id: int, run_id: int, is_warmup: bool) -> BenchResult:
        async with sem:
            if framework == "RawLMStudio":
                return await bench_raw_lmstudio(
                    prompt, prompt_id, run_id, is_warmup, args.lm_studio_url, args.local_model, args.timeout_s
                )
            if framework == "DeepAgents":
                return await bench_deepagents_endpoint(prompt, prompt_id, run_id, is_warmup, args.api_base_url, args.timeout_s)
            if framework == "ClaudeAgentSDK":
                return await bench_claude_sdk_endpoint(prompt, prompt_id, run_id, is_warmup, args.api_base_url, args.timeout_s)
            raise ValueError(f"Unknown framework: {framework}")

    frameworks: list[str] = []
    if not args.skip_raw:
        frameworks.append("RawLMStudio")
    if not args.skip_deepagents:
        frameworks.append("DeepAgents")
    if not args.skip_claude:
        frameworks.append("ClaudeAgentSDK")

    if args.concurrency == 1:
        for round_num in range(1, args.rounds + 1):
            print(f"\n── Round {round_num}/{args.rounds} ──")
            for prompt_id, prompt in enumerate(PROMPTS):
                short = prompt[:50]
                print(f"\n  Prompt: \"{short}...\"")
                for fw_i, fw in enumerate(frameworks, start=1):
                    label = f"[{fw_i}/{len(frameworks)}] {fw}..."
                    print(f"    {label}")
                    is_warmup = round_num <= args.warmup
                    r = await _run_one(fw, prompt, prompt_id, round_num, is_warmup)
                    results.append(r)
                    if r.error:
                        print(f"          ERROR: {r.error}")
                    else:
                        warm = " (warmup)" if r.is_warmup else ""
                        print(f"          TTFT={_fmt_num(r.time_to_first_token_s)}s  total={_fmt_num(r.total_time_s)}s  tok/s={r.tokens_per_second or 'n/a'}{warm}")
    else:
        total_jobs = args.rounds * len(PROMPTS) * len(frameworks)
        print(f"\nRunning {total_jobs} requests with concurrency={args.concurrency}...")
        jobs: list[asyncio.Task[BenchResult]] = []
        for round_num in range(1, args.rounds + 1):
            is_warmup = round_num <= args.warmup
            for prompt_id, prompt in enumerate(PROMPTS):
                for fw in frameworks:
                    jobs.append(asyncio.create_task(_run_one(fw, prompt, prompt_id, round_num, is_warmup)))
        results.extend(await asyncio.gather(*jobs))

    return results


def summarize(results: list[BenchResult]) -> None:
    """Print per-framework percentiles + baseline deltas."""
    from collections import defaultdict
    buckets: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        if r.is_warmup:
            continue
        buckets[r.framework].append(r)

    raw_index: dict[tuple[int, int], BenchResult] = {}
    for r in buckets.get("RawLMStudio", []):
        if r.error:
            continue
        raw_index[(r.prompt_id, r.run_id)] = r

    print("\n\n════════════════════════════════════")
    print("  SUMMARY  (p50 / p90 / p95, warmups excluded)")
    print("════════════════════════════════════\n")

    order = ["RawLMStudio", "DeepAgents", "ClaudeAgentSDK"]
    for fw in order:
        items = buckets.get(fw, [])
        if not items:
            continue
        good = [i for i in items if not i.error]
        if not good:
            print(f"- {fw}: all runs errored.")
            continue

        ttfts = [i.time_to_first_token_s for i in good if i.time_to_first_token_s is not None]
        totals = [i.total_time_s for i in good]
        tps_vals = [i.tokens_per_second for i in good if i.tokens_per_second is not None]
        err_rate = 1.0 - (len(good) / max(1, len(items)))

        # Baseline deltas (if raw baseline exists for matching (prompt_id, run_id))
        d_ttft: list[float] = []
        d_total: list[float] = []
        if fw != "RawLMStudio" and raw_index:
            for i in good:
                raw = raw_index.get((i.prompt_id, i.run_id))
                if not raw:
                    continue
                if i.time_to_first_token_s is not None and raw.time_to_first_token_s is not None:
                    d_ttft.append(i.time_to_first_token_s - raw.time_to_first_token_s)
                d_total.append(i.total_time_s - raw.total_time_s)

        print(f"{fw}  (n={len(items)}, ok={len(good)}, err={err_rate*100:.1f}%)")
        print(
            f"  TTFT  p50={_fmt_num(_percentile(ttfts, 50))}  p90={_fmt_num(_percentile(ttfts, 90))}  p95={_fmt_num(_percentile(ttfts, 95))}"
        )
        print(
            f"  Total p50={_fmt_num(_percentile(totals, 50))}  p90={_fmt_num(_percentile(totals, 90))}  p95={_fmt_num(_percentile(totals, 95))}"
        )
        if tps_vals:
            print(f"  Tok/s avg={_fmt_num(sum(tps_vals)/len(tps_vals), digits=1)}")
        if d_total:
            print(
                f"  ΔTTFT p50={_fmt_num(_percentile(d_ttft, 50))}  ΔTotal p50={_fmt_num(_percentile(d_total, 50))}  (vs RawLMStudio)"
            )
        print("")

    # Save JSON
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results saved to: {output_path}")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Agent framework latency benchmark")
    p.add_argument("--api-base-url", default="http://127.0.0.1:8001",
                   help="FastAPI base URL (default: http://127.0.0.1:8001)")
    p.add_argument("--lm-studio-url", default="http://localhost:1234",
                   help="LM Studio server URL (default: http://localhost:1234)")
    p.add_argument("--local-model", default="local-model",
                   help="Model name as shown in LM Studio (default: 'local-model')")
    p.add_argument("--timeout-s", type=float, default=180,
                   help="Per-request timeout seconds (default: 180)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup rounds (excluded from summary) (default: 1)")
    p.add_argument("--rounds", type=int, default=3,
                   help="Number of benchmark rounds per prompt (default: 3)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent requests (default: 1)")
    p.add_argument("--skip-raw", action="store_true",
                   help="Skip the raw LM Studio baseline")
    p.add_argument("--skip-deepagents", action="store_true",
                   help="Skip LangChain Deep Agents benchmark")
    p.add_argument("--skip-claude", action="store_true",
                   help="Skip Claude Agent SDK benchmark")
    return p.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    print("╔══════════════════════════════════════════════╗")
    print("║  Agent Framework Latency Benchmark           ║")
    print("║  LangChain DeepAgents vs Claude Agent SDK    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nFastAPI URL    : {args.api_base_url}")
    print(f"\nLM Studio URL  : {args.lm_studio_url}")
    print(f"Local model    : {args.local_model}")
    print(f"Rounds         : {args.rounds}")
    print(f"Warmup rounds  : {args.warmup}")
    print(f"Concurrency    : {args.concurrency}")
    print(f"Timeout (s)    : {args.timeout_s}")

    results = asyncio.run(run_all(args))
    summarize(results)
