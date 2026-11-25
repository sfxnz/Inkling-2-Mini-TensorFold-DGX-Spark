"""Cold and warm server startup timing until GET /v1/models returns 200 OK with loaded models.

Examples:
  python3 tools/startup.py --mode measure --url http://127.0.0.1:8000
  python3 tools/startup.py --mode warm --out runs/startup-warm.json
  python3 tools/startup.py --mode both --out runs/startup.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def wait_models(url, timeout_s=900, container_name=None):
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    models_url = url.rstrip("/") + "/v1/models"
    last_err = None
    while time.monotonic() < deadline:
        if container_name:
            st = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True,
            )
            if st.returncode == 0 and st.stdout.strip() != "true":
                return None, f"container {container_name} exited before /v1/models became ready"
        try:
            status, body = recipe.get(models_url, timeout=5)
            if status == 200 and body.get("data"):
                return round(time.monotonic() - t0, 3), None
            last_err = f"status={status}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(0.25)
    return None, f"timeout after {timeout_s}s ({last_err})"


def extract_phases(container_name):
    if not container_name:
        return []
    res = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True)
    lines = (res.stdout + "\n" + res.stderr).splitlines()
    interesting = []
    pat = re.compile(r"(load|pack|graph|kv|pool|ready|serving|tensorfold|draft|cache|error|warn)", re.I)
    for ln in lines:
        if pat.search(ln):
            interesting.append(ln)
    return interesting[-40:]


def run_cycle(mode, url, timeout_s, container_name):
    env = os.environ.copy()
    cold_vol = None
    if mode == "cold":
        cold_vol = f"tf-inkling2-coldtest-{int(time.time())}"
        subprocess.run(["docker", "volume", "create", cold_vol], check=True, capture_output=True)
        env["ENGINE_CACHE"] = cold_vol

    try:
        subprocess.run([str(ROOT / "stop.sh")], env=env, check=True)
        t0 = time.monotonic()
        # Launch container directly via run.sh? run.sh waits for /health internally; time the full run.sh
        # and then verify /v1/models 200 OK immediately after (or time from process start until /v1/models 200).
        proc = subprocess.Popen([str(ROOT / "run.sh")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ready_s, err = wait_models(url, timeout_s=timeout_s, container_name=None)
        wall_total = round(time.monotonic() - t0, 3)
        rc = proc.wait(timeout=60)
        out_log = proc.stdout.read() if proc.stdout else ""
        if rc != 0 and ready_s is None:
            err = err or f"run.sh exited {rc}"
        return {
            "mode": mode,
            "ok": err is None,
            "models_200_s": ready_s,
            "wall_s": wall_total,
            "error": err,
            "phases": extract_phases(container_name),
            "run_sh_tail": out_log.splitlines()[-20:],
        }
    finally:
        if cold_vol:
            subprocess.run([str(ROOT / "stop.sh")], env=env, capture_output=True)
            subprocess.run(["docker", "volume", "rm", "-f", cold_vol], capture_output=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["measure", "warm", "cold", "both"], default="measure")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--container", default=recipe.setting("CONTAINER_NAME", "tf-inkling2"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", help="write JSON report here")
    a = ap.parse_args(argv)

    results = []
    if a.mode == "measure":
        ready_s, err = wait_models(a.url, timeout_s=a.timeout, container_name=a.container)
        results.append({
            "mode": "measure",
            "ok": err is None,
            "models_200_s": ready_s,
            "error": err,
            "phases": extract_phases(a.container),
        })
    elif a.mode == "both":
        results.append(run_cycle("cold", a.url, a.timeout, a.container))
        results.append(run_cycle("warm", a.url, a.timeout, a.container))
    else:
        results.append(run_cycle(a.mode, a.url, a.timeout, a.container))

    for r in results:
        print(f"startup mode={r['mode']} ok={r['ok']} models_200_s={r['models_200_s']} err={r['error']}", flush=True)

    report = {
        "url": a.url,
        "container": a.container,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
