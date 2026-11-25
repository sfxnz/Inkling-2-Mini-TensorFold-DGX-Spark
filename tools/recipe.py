"""Settings from recipe.conf (environment overrides) and shared HTTP helpers for Inkling-2-Mini tools."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

CONF = Path(__file__).resolve().parent.parent / "recipe.conf"


def setting(name, default=None):
    if name in os.environ:
        return os.environ[name]
    if CONF.is_file():
        text = CONF.read_text()
        m = re.search(rf"^{name}=\$\{{{name}:-(.*)\}}$", text, re.M)
        if m:
            val = m.group(1)
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            return val
        m2 = re.search(rf"^{name}=(.*)$", text, re.M)
        if m2:
            val = m2.group(1).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            return val
    if default is not None:
        return default
    raise KeyError(f"{name} has no default in {CONF}")


def default_url():
    return f"http://127.0.0.1:{setting('PORT', '8000')}"


def headers():
    h = {"content-type": "application/json"}
    if os.environ.get("API_KEY"):
        h.update({"authorization": f"Bearer {os.environ['API_KEY']}", "x-api-key": os.environ["API_KEY"]})
    return h


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": {"message": raw.decode(errors="replace")}}


def post(url, body, timeout=900):
    """Return (status, parsed JSON body); HTTP errors return status and error body instead of raising."""
    req = urllib.request.Request(url, json.dumps(body).encode(), headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": {"message": raw.decode(errors="replace")}}


def stream(url, body, timeout=900):
    """Yield (seconds since request start, parsed SSE JSON chunk)."""
    req = urllib.request.Request(url, json.dumps({**body, "stream": True}).encode(), headers())
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                yield time.monotonic() - t0, json.loads(line[5:])
