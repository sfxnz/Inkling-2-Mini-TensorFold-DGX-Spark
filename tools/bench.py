"""Two-ruler benchmark for Inkling-2-Mini on TensorFold.

Rulers:
  inner: Deterministic greedy (temperature=0, thinking=False, >=6 samples) reporting
         ms/round, tokens/round, acceptance_rate, decode_tok_s, aggregate_tok_s, ttft_s.
  outer: Honest sampled prose (thinking=False, default sampling) plus community-comparable
         cells (thinking=True temp=0.7 top_k=20, code, greedy thinking=False, c1/c2/c4/c8).
  all:   Runs both inner and outer rulers and writes one combined JSON report.

Examples:
  python3 tools/bench.py --ruler inner --concurrency 1,4,8 --runs 6 --out runs/inner.json
  python3 tools/bench.py --ruler outer --out runs/outer.json
  python3 tools/bench.py --concurrency 1,4 --runs 6 --temperature 0 --prompts prose
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

PROSE_PROMPTS = [
    "Write a detailed 600-word essay about the history of the printing press.",
    "Explain, step by step and at length, how a CPU pipeline handles branch misprediction.",
    "Write a long short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe in detail the water cycle and its effect on regional climates.",
    "Write a long account of how the transcontinental railroads changed trade in the nineteenth century.",
    "Explain at length how vaccines train the immune system, with examples from history.",
    "Write a detailed travel guide to a fictional coastal city, district by district.",
    "Describe at length how a modern compiler turns source code into machine code.",
]

CODE_PROMPTS = [
    "Write a complete Python module that implements an LRU cache with TTL support, with type hints, docstrings and unit tests.",
    "Write a C program that implements a hash map with open addressing: insert, lookup, delete and resize, with comments.",
    "Write a Rust implementation of a binary min-heap with push, pop, peek and heapify, plus its tests.",
    "Write a TypeScript module that parses and evaluates arithmetic expressions with a recursive descent parser, with tests.",
    "Write a Go HTTP server with JSON CRUD endpoints for a todo list, in-memory storage and logging middleware.",
    "Write a SQL schema for a library management system and twenty example queries against it.",
    "Write a Bash script that backs up a directory with rotation, logging and error handling, with comments.",
    "Write a Python script that reads a CSV file, computes per-column statistics and writes a Markdown report, with argparse.",
]

PROMPT_SETS = {"prose": PROSE_PROMPTS, "code": CODE_PROMPTS}


def looping(text, n=8, thresh=0.5):
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n))]
    return len(grams) > 50 and (1.0 - len(set(grams)) / len(grams)) > thresh


def one_stream(chat_url, model, prompt, max_tokens, thinking=False, temperature=None, top_k=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "ignore_eos": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_k is not None:
        body["top_k"] = top_k

    first = last = None
    deltas, usage, spec, end = 0, None, None, 0.0
    chunks_text = []
    for t, chunk in recipe.stream(chat_url, body):
        for c in chunk.get("choices", []):
            d = c.get("delta") or {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
            if piece:
                first = t if first is None else first
                last = t
                deltas += 1
                chunks_text.append(piece)
        usage = chunk.get("usage") or usage
        spec = chunk.get("speculative") or spec
        end = t

    tokens = usage["completion_tokens"] if usage else deltas
    decode_s = (last - first) if (first is not None and last is not None and last > first) else None
    decode_tok_s = round((tokens - 1) / decode_s, 2) if decode_s and tokens > 1 else None

    rounds = spec.get("rounds") if isinstance(spec, dict) else None
    drafted = spec.get("drafted") if isinstance(spec, dict) else None
    accepted = spec.get("accepted") if isinstance(spec, dict) else None
    acc_rate = spec.get("acceptance_rate") if isinstance(spec, dict) else None
    tpr = spec.get("tokens_per_round") if isinstance(spec, dict) else None
    ms_per_round = None
    if decode_s and rounds:
        decode_rounds = (rounds - 1) if rounds > 1 else rounds
        ms_per_round = round((decode_s * 1000.0) / decode_rounds, 3)
        if tpr is None:
            tpr = round(tokens / rounds, 3)

    return {
        "ttft_s": round(first, 3) if first is not None else None,
        "tokens": tokens,
        "tokens_from": "usage" if usage else "content deltas",
        "decode_tok_s": decode_tok_s,
        "wall_s": round(end, 3),
        "rounds": rounds,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": round(acc_rate, 4) if acc_rate is not None else None,
        "tokens_per_round": round(tpr, 3) if tpr is not None else None,
        "ms_per_round": ms_per_round,
        "loop": looping("".join(chunks_text)),
        "speculative": spec,
    }


def run_once(chat_url, model, concurrency, max_tokens, thinking=False, temperature=None, top_k=None, prompts=PROSE_PROMPTS):
    out = [None] * concurrency

    def worker(i):
        try:
            out[i] = one_stream(
                chat_url, model, prompts[i % len(prompts)], max_tokens,
                thinking=thinking, temperature=temperature, top_k=top_k,
            )
        except Exception as e:
            out[i] = {"error": repr(e)}

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    good = [s for s in out if "error" not in s]
    return {
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "errors": concurrency - len(good),
        "loops": sum(1 for s in good if s.get("loop")),
        "aggregate_tok_s": round(sum(s["tokens"] for s in good) / wall, 2) if wall > 0 else None,
        "streams": out,
    }


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def summarize(runs):
    streams = [s for r in runs for s in r["streams"] if "error" not in s]
    return {
        "concurrency": runs[0]["concurrency"],
        "runs": len(runs),
        "errors": sum(r["errors"] for r in runs),
        "loops": sum(r["loops"] for r in runs),
        "median_ttft_s": med(s["ttft_s"] for s in streams),
        "median_decode_tok_s": med(s["decode_tok_s"] for s in streams),
        "median_aggregate_tok_s": med(r["aggregate_tok_s"] for r in runs),
        "median_ms_per_round": med(s["ms_per_round"] for s in streams),
        "median_tokens_per_round": med(s["tokens_per_round"] for s in streams),
        "median_acceptance_rate": med(s["acceptance_rate"] for s in streams),
    }


def run_cell(chat_url, model, cell_name, levels, runs_n, warmup_n, max_tokens, thinking, temperature, top_k, prompt_set):
    prompts = PROMPT_SETS[prompt_set]
    print(f"\n=== cell: {cell_name} (prompts={prompt_set} thinking={thinking} temp={temperature} top_k={top_k} runs={runs_n}) ===")
    print(f"{'c':>3} {'runs':>4} {'err':>3} {'ttft_s':>8} {'decode':>9} {'agg':>9} {'ms/rnd':>8} {'tok/rnd':>8} {'acc':>7}")
    cell_runs = []
    cell_summary = []
    for c in levels:
        for _ in range(warmup_n):
            run_once(chat_url, model, c, max_tokens, thinking, temperature, top_k, prompts)
        runs = [run_once(chat_url, model, c, max_tokens, thinking, temperature, top_k, prompts) for _ in range(runs_n)]
        s = summarize(runs)
        cell_runs.extend(runs)
        cell_summary.append(s)
        print(
            f"{c:>3} {s['runs']:>4} {s['errors']:>3} {s['median_ttft_s']!s:>8} "
            f"{s['median_decode_tok_s']!s:>9} {s['median_aggregate_tok_s']!s:>9} "
            f"{s['median_ms_per_round']!s:>8} {s['median_tokens_per_round']!s:>8} "
            f"{s['median_acceptance_rate']!s:>7}",
            flush=True,
        )
    return {
        "cell": cell_name,
        "settings": {
            "concurrency": levels,
            "runs": runs_n,
            "warmup": warmup_n,
            "max_tokens": max_tokens,
            "thinking": thinking,
            "temperature": temperature,
            "top_k": top_k,
            "prompts": prompt_set,
        },
        "summary": cell_summary,
        "runs": cell_runs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruler", choices=["custom", "inner", "outer", "all"], default="custom")
    ap.add_argument("--concurrency", default="1,4,8", help="comma list, e.g. 1,4,8")
    ap.add_argument("--runs", type=int, default=6, help="timed runs per concurrency level")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup runs per level")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, help="sent only when set; server default otherwise")
    ap.add_argument("--top-k", type=int, help="sent only when set")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prompts", choices=sorted(PROMPT_SETS), default="prose")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="write JSON report here")
    a = ap.parse_args(argv)
    chat_url = a.url.rstrip("/") + "/v1/chat/completions"
    levels = [int(c) for c in a.concurrency.split(",") if c.strip()]

    cells = []
    if a.ruler == "custom":
        cells.append(run_cell(
            chat_url, a.model, "custom", levels, a.runs, a.warmup,
            a.max_tokens, a.thinking, a.temperature, a.top_k, a.prompts,
        ))
    else:
        if a.ruler in ("inner", "all"):
            cells.append(run_cell(
                chat_url, a.model, "inner_greedy_prose", levels, max(a.runs, 6), a.warmup,
                a.max_tokens, False, 0.0, None, "prose",
            ))
        if a.ruler in ("outer", "all"):
            outer_levels = [int(c) for c in a.concurrency.split(",") if c.strip()] if a.concurrency != "1,4,8" else [1, 2, 4, 8]
            cells.append(run_cell(
                chat_url, a.model, "outer_sampled_prose_nothink", outer_levels, a.runs, a.warmup,
                a.max_tokens, False, None, None, "prose",
            ))
            cells.append(run_cell(
                chat_url, a.model, "outer_thinking_on_prose", outer_levels, a.runs, a.warmup,
                a.max_tokens, True, 0.7, 20, "prose",
            ))
            cells.append(run_cell(
                chat_url, a.model, "outer_code_nothink", outer_levels, a.runs, a.warmup,
                a.max_tokens, False, None, None, "code",
            ))
            cells.append(run_cell(
                chat_url, a.model, "outer_greedy_prose_nothink", outer_levels, a.runs, a.warmup,
                a.max_tokens, False, 0.0, None, "prose",
            ))

    report = {
        "ruler": a.ruler,
        "url": a.url,
        "model": a.model,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": cells,
        "summary": cells[0]["summary"] if len(cells) == 1 else [
            {"cell": c["cell"], "summary": c["summary"]} for c in cells
        ],
    }
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")

    any_err = any(s["errors"] > 0 for c in cells for s in c["summary"])
    return 1 if any_err else 0


if __name__ == "__main__":
    sys.exit(main())
