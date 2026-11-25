"""Cold prefill benchmark: unique prompts at 4K..128K verifying cached_tokens == 0, reporting tok/s and TTFT.

python3 tools/prefill.py [--lengths 4096,8192,16384,32768,65536,131072] [--runs 1] [--seed N] [--out prefill.json]
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

CITIES = ["Lisbon", "Osaka", "Quito", "Tromso", "Nairobi", "Hobart", "Tbilisi", "Cusco", "Gdansk", "Leeds"]
ITEMS = ["lanterns", "barrels", "ledgers", "copper pipes", "wool bales", "clock gears", "maps", "seed sacks"]


def records(n, seed=7):
    rng = random.Random(seed)
    return [
        f"Record {i}: warehouse {rng.choice(CITIES)} logged {rng.randint(10, 9999)} {rng.choice(ITEMS)} "
        f"on shelf {rng.randint(1, 400)} shift {rng.randint(1, 3)}.\n"
        for i in range(n)
    ]


def one(chat_url, model, text):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first, usage = None, {}
    for t, chunk in recipe.stream(chat_url, body, timeout=7200):
        if first is None and any(
            (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
            for c in chunk.get("choices", [])
        ):
            first = t
        usage = chunk.get("usage") or usage
    pt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cold_exact = (cached == 0)
    return {
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "cold_exact": cold_exact,
        "ttft_s": round(first, 3) if first is not None else None,
        "prefill_tok_s": round(pt / first, 1) if first and pt else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lengths_pos", nargs="?", help="optional positional comma list of lengths")
    ap.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072",
                    help="comma list of prompt lengths (default 4K..128K)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=time.time_ns() % 1000000000,
                    help="unique seed defeating prefix caching")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--allow-template-cache", action="store_true",
                    help="allow <1%% cached_tokens for tiny shared chat-template prefix instead of strict cached_tokens==0")
    ap.add_argument("--out", help="write JSON report here")
    a = ap.parse_args(argv)

    raw_lengths = a.lengths_pos or a.lengths
    targets = [int(x) for x in raw_lengths.split(",") if x.strip()]
    chat_url = a.url.rstrip("/") + "/v1/chat/completions"

    probe = records(400, a.seed)
    status, d = recipe.post(chat_url, {
        "model": a.model,
        "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": f"Probe {a.seed}.\n" + "".join(probe)}],
    })
    if status != 200:
        print(json.dumps({"event": "error", "stage": "calibration", "status": status, "body": d}))
        return 1
    per_line = d["usage"]["prompt_tokens"] / len(probe)

    report = {
        "settings": {
            "url": a.url,
            "model": a.model,
            "seed": a.seed,
            "runs": a.runs,
            "lengths": targets,
            "strict_zero_cache": not a.allow_template_cache,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "runs": [],
        "summary": [],
    }

    print(f"{'target':>8} {'prompt_tok':>10} {'cached':>7} {'cold':>5} {'ttft_s':>8} {'prefill tok/s':>14}")
    failed = False
    for target in targets:
        runs = []
        for r in range(a.runs):
            n_lines = max(1, int((target - 16) / per_line))
            lines = records(n_lines, a.seed + 10000 * r + target)
            text = f"UniqueBatch-{a.seed}-{target}-{r}-{time.time_ns()}.\n" + "".join(lines) + "\nReply OK."
            try:
                run = one(chat_url, a.model, text)
            except Exception as e:
                run = {"error": repr(e), "cold_exact": False, "cached_tokens": None}
                failed = True
            cached = run.get("cached_tokens")
            pt = run.get("prompt_tokens") or 0
            is_cold = (cached == 0) if not a.allow_template_cache else ((cached or 0) * 100 < pt)
            run["cold"] = is_cold
            if not is_cold:
                failed = True
            runs.append({"target": target, "run": r, **run})
            print(
                f"{target:>8} {pt!s:>10} {cached!s:>7} {str(is_cold):>5} "
                f"{run.get('ttft_s')!s:>8} {run.get('prefill_tok_s')!s:>14}",
                flush=True,
            )
        cold = [x for x in runs if x.get("cold") and x.get("prefill_tok_s")]
        report["runs"].extend(runs)
        report["summary"].append({
            "target": target,
            "cold_runs": len(cold),
            "median_prompt_tokens": statistics.median(x["prompt_tokens"] for x in cold) if cold else None,
            "median_ttft_s": round(statistics.median(x["ttft_s"] for x in cold), 3) if cold else None,
            "median_prefill_tok_s": round(statistics.median(x["prefill_tok_s"] for x in cold), 1) if cold else None,
        })

    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed or any(s["cold_runs"] == 0 for s in report["summary"]) else 0


if __name__ == "__main__":
    sys.exit(main())
