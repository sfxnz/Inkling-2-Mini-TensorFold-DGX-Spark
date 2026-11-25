"""Acceptance gates for Inkling-2-Mini on TensorFold.

Checks:
  load, chat, reasoning, tools, structured, vision, drafted_exactness,
  multilingual, prefix_cache, needle_256k, needle_1m, soak

Usage:
  python3 tools/gates.py [--url http://127.0.0.1:8000] [--plain-url http://127.0.0.1:8001]
                         [--only load,chat,reasoning] [--skip needle_1m] [--out gates.json]
"""
import argparse
import base64
import io
import json
import random
import re
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

QUESTION = "What is 17 * 23? Reply with the number only."
PROSE = [
    "Write a detailed paragraph about the history of the printing press.",
    "Explain how a CPU pipeline handles a branch misprediction.",
    "Describe the water cycle and its effect on regional climates.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how hash tables resolve collisions using open addressing.",
    "Describe the process of cellular respiration in eukaryotic cells.",
    "Summarize the architectural differences between RISC and CISC processors.",
    "Explain how public-key cryptography enables secure communication.",
]

CITIES = ["Lisbon", "Osaka", "Quito", "Tromso", "Nairobi", "Hobart", "Tbilisi", "Cusco", "Gdansk", "Leeds"]
ITEMS = ["lanterns", "barrels", "ledgers", "copper pipes", "wool bales", "clock gears", "maps", "seed sacks"]
WORDS = ["ALPHA", "BRAVO", "CEDAR", "DELTA", "EMBER", "FJORD", "GAMMA", "HAZEL"]


def shapes_png_fallback(w=256, h=128):
    rows = []
    for y in range(h):
        row = bytearray(b"\x00")
        for x in range(w):
            if 24 <= x < 104 and 24 <= y < 104:
                row += b"\xff\x00\x00"
            elif (x - 192) ** 2 + (y - 64) ** 2 <= 40 ** 2:
                row += b"\x00\x00\xff"
            else:
                row += b"\xff\xff\xff"
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def make_shapes_image(fmt, w=256, h=128):
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([24, 24, 103, 103], fill=(255, 0, 0))
        draw.ellipse([152, 24, 232, 104], fill=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format=fmt.upper())
        return buf.getvalue()
    except ImportError:
        if fmt.lower() == "png":
            return shapes_png_fallback(w, h)
        raise


def records(n, seed=7):
    rng = random.Random(seed)
    return [
        f"Record {i}: warehouse {rng.choice(CITIES)} logged {rng.randint(10, 9999)} {rng.choice(ITEMS)} "
        f"on shelf {rng.randint(1, 400)} shift {rng.randint(1, 3)}.\n"
        for i in range(n)
    ]


def looping(text, n=8, thresh=0.5):
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n))]
    return len(grams) > 50 and (1.0 - len(set(grams)) / len(grams)) > thresh


class Gates:
    def __init__(self, url, model, context, parallel, plain_url=None, allow_rescale=False):
        self.url = url.rstrip("/")
        self.chat_url = self.url + "/v1/chat/completions"
        self.model = model
        self.context = context
        self.parallel = parallel
        self.plain_url = plain_url.rstrip("/") if plain_url else None
        self.allow_rescale = allow_rescale
        self.served_window = None

    def chat(self, messages, thinking=False, url=None, timeout=900, **extra):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.model, "messages": messages, **extra}
        body.setdefault("chat_template_kwargs", {"enable_thinking": thinking})
        return recipe.post((url or self.url) + "/v1/chat/completions", body, timeout=timeout)

    def gate_load(self):
        st_h, health = recipe.get(self.url + "/health")
        st_m, models = recipe.get(self.url + "/v1/models")
        ours = [m for m in models.get("data", []) if m.get("id") == self.model]
        self.served_window = ours[0].get("context_length") if ours else None
        ok = (st_h == 200 and st_m == 200 and bool(ours) and self.served_window == self.context)
        return ok, {"health": st_h, "models_status": st_m, "want_context": self.context,
                    "got_context": self.served_window, "health_body": health}

    def gate_chat(self):
        st, d = self.chat("Say the exact phrase: INKLING_OK", thinking=False, max_tokens=64, temperature=0)
        msg = (d.get("choices") or [{}])[0].get("message", {}) if st == 200 else {}
        content = msg.get("content") or ""
        ok = st == 200 and "INKLING_OK" in content
        return ok, {"status": st, "content": content}

    def gate_reasoning(self):
        st1, d1 = self.chat(QUESTION, thinking=True, max_tokens=4096, temperature=0)
        m1 = (d1.get("choices") or [{}])[0].get("message", {}) if st1 == 200 else {}
        r1, c1 = m1.get("reasoning_content") or "", m1.get("content") or ""
        ok_on = (st1 == 200 and bool(r1.strip()) and "391" in c1
                 and "<think>" not in c1 and "</think>" not in c1)

        st2, d2 = self.chat(QUESTION, thinking=False, max_tokens=64, temperature=0)
        m2 = (d2.get("choices") or [{}])[0].get("message", {}) if st2 == 200 else {}
        r2, c2 = m2.get("reasoning_content") or "", m2.get("content") or ""
        ok_off = (st2 == 200 and not r2 and "391" in c2
                  and "<think>" not in c2 and "</think>" not in c2)
        return ok_on and ok_off, {
            "thinking_on": {"ok": ok_on, "reasoning_chars": len(r1), "content": c1},
            "thinking_off": {"ok": ok_off, "reasoning_chars": len(r2), "content": c2},
        }

    def gate_tools(self):
        schema = {
            "type": "object",
            "required": ["city", "unit"],
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        }
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get weather for a city", "parameters": schema,
        }}]
        msgs = [{"role": "user", "content": "Look up the weather in Paris, France in celsius. Use get_weather."}]
        st1, d1 = self.chat(msgs, thinking=False, max_tokens=256, temperature=0, tools=tools, tool_choice="required")
        msg1 = (d1.get("choices") or [{}])[0].get("message", {}) if st1 == 200 else {}
        calls = msg1.get("tool_calls") or []
        if not calls:
            return False, {"stage": "turn1", "status": st1, "body": d1}
        call = calls[0]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception:
            args = {}
        ok_call = (call["function"]["name"] == "get_weather" and "city" in args and "unit" in args)

        msgs.append(msg1)
        msgs.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps({"city": "Paris", "temperature_c": 19, "condition": "sunny"}),
        })
        st2, d2 = self.chat(msgs, thinking=False, max_tokens=128, temperature=0, tools=tools)
        c2 = ((d2.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "" if st2 == 200 else ""
        ok_reply = st2 == 200 and ("19" in c2 or "sunny" in c2.lower())
        return ok_call and ok_reply, {"call_args": args, "turn2_status": st2, "turn2_content": c2}

    def gate_structured(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["city", "population_millions"],
            "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
        }
        st1, d1 = self.chat(
            "Return JSON with city='Tokyo' and population_millions=14.",
            thinking=False, max_tokens=128, temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "city_info", "strict": True, "schema": schema}},
        )
        c1 = ((d1.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "" if st1 == 200 else ""
        try:
            j1 = json.loads(c1)
            ok_schema = isinstance(j1, dict) and "city" in j1 and isinstance(j1.get("population_millions"), (int, float))
        except Exception:
            j1, ok_schema = None, False

        st2, d2 = self.chat(
            'Reply with a JSON object having keys "ok": true and "count": 3.',
            thinking=False, max_tokens=128, temperature=0,
            response_format={"type": "json_object"},
        )
        c2 = ((d2.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "" if st2 == 200 else ""
        try:
            j2 = json.loads(c2)
            ok_obj = isinstance(j2, dict)
        except Exception:
            j2, ok_obj = None, False

        tools = [
            {"type": "function", "function": {
                "name": "pick_number", "description": "Pick a number",
                "parameters": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}},
            }},
            {"type": "function", "function": {
                "name": "other_fn", "description": "Other",
                "parameters": {"type": "object", "properties": {}},
            }},
        ]
        st3, d3 = self.chat(
            "Call pick_number with n=42.", thinking=False, max_tokens=128, temperature=0,
            tools=tools, tool_choice={"type": "function", "function": {"name": "pick_number"}},
        )
        calls3 = (((d3.get("choices") or [{}])[0].get("message", {}) or {}).get("tool_calls") or []) if st3 == 200 else []
        ok_tc = bool(calls3) and calls3[0]["function"]["name"] == "pick_number"
        return ok_schema and ok_obj and ok_tc, {
            "json_schema": {"ok": ok_schema, "parsed": j1, "status": st1},
            "json_object": {"ok": ok_obj, "parsed": j2, "status": st2},
            "tool_choice": {"ok": ok_tc, "calls": calls3, "status": st3},
        }

    def gate_vision(self):
        formats = [("png", "image/png"), ("jpeg", "image/jpeg"), ("webp", "image/webp")]
        prompt = ('The image shows two shapes. Reply with JSON only: '
                  '{"left": {"color": "...", "shape": "..."}, "right": {"color": "...", "shape": "..."}}')
        details = {}
        all_ok = True
        for fmt, mime in formats:
            raw = make_shapes_image(fmt)
            url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
            st, d = self.chat(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                              {"type": "text", "text": prompt}]}],
                thinking=False, max_tokens=200, temperature=0,
            )
            content = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "" if st == 200 else ""
            m = re.search(r"\{.*\}", content, re.S)
            try:
                got = json.loads(m.group(0)) if m else {}
                ok = ("red" in got["left"]["color"].lower() and "square" in got["left"]["shape"].lower()
                      and "blue" in got["right"]["color"].lower() and "circle" in got["right"]["shape"].lower())
            except Exception:
                ok = False
            details[fmt] = {"ok": ok, "status": st, "content": content}
            all_ok = all_ok and ok
        return all_ok, details

    def gate_drafted_exactness(self):
        prompts = PROSE[:4]
        serial_results = []
        for p in prompts:
            st, d = self.chat(p, thinking=False, max_tokens=256, temperature=0, seed=42)
            msg = (d.get("choices") or [{}])[0].get("message", {}) if st == 200 else {}
            spec = d.get("speculative") or {}
            serial_results.append({
                "status": st,
                "content": msg.get("content") or "",
                "drafted": spec.get("drafted", 0),
                "accepted": spec.get("accepted", 0),
            })

        conc_results = [None] * len(prompts)

        def worker(i):
            st, d = self.chat(prompts[i], thinking=False, max_tokens=256, temperature=0, seed=42)
            msg = (d.get("choices") or [{}])[0].get("message", {}) if st == 200 else {}
            conc_results[i] = {"status": st, "content": msg.get("content") or "", "speculative": d.get("speculative") or {}}

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        serial_vs_conc = [s["content"] == c["content"] and s["status"] == 200 and c["status"] == 200
                          for s, c in zip(serial_results, conc_results)]
        drafted_active = all(s["drafted"] > 0 for s in serial_results)

        plain_matches = None
        if self.plain_url:
            plain_matches = []
            for i, p in enumerate(prompts):
                stp, dp = self.chat(p, thinking=False, max_tokens=256, temperature=0, seed=42, url=self.plain_url)
                cp = (((dp.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "") if stp == 200 else ""
                plain_matches.append(cp == serial_results[i]["content"] and cp == conc_results[i]["content"] and stp == 200)

        ok = all(serial_vs_conc) and (all(plain_matches) if plain_matches is not None else True)
        return ok, {
            "drafted_active": drafted_active,
            "serial_vs_concurrent_exact": serial_vs_conc,
            "plain_url": self.plain_url,
            "plain_vs_drafted_exact": plain_matches,
        }

    def gate_multilingual(self):
        probes = [
            ("ko", "다음 한국어 문장을 간단히 요약하세요: 인공지능 기술은 빠르게 발전하고 있습니다."),
            ("ja", "次の日本語の文を英語に翻訳してください: 今日はとても良い天気です。"),
            ("zh", "请用一句话解释什么是分布式系统。"),
            ("ar", "اشرح باختصار معنى الذكاء الاصطناعي."),
            ("emoji", "Reply with one sentence explaining what 🚀 and 🔍 mean in software releases."),
        ]
        results = {}
        all_ok = True
        for lang, prompt in probes:
            st, d = self.chat(prompt, thinking=False, max_tokens=160, temperature=0)
            c = (((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "") if st == 200 else ""
            has_fffd = "\ufffd" in c
            ok = st == 200 and len(c.strip()) > 0 and not has_fffd
            results[lang] = {"ok": ok, "status": st, "has_fffd": has_fffd, "len": len(c), "preview": c[:120]}
            all_ok = all_ok and ok
        return all_ok, results

    def gate_prefix_cache(self):
        salt = f"PrefixCacheSalt-{time.time_ns()}"
        prefix = f"[{salt}]\n" + "".join(records(350, seed=123))
        st1, d1 = self.chat(prefix + "\nSummarize the records above in one sentence.", thinking=False, max_tokens=64, temperature=0)
        st2, d2 = self.chat(prefix + "\nList any two cities mentioned in the records above.", thinking=False, max_tokens=64, temperature=0)
        c1 = ((d1.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") if st1 == 200 else None
        c2 = ((d2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") if st2 == 200 else None
        ok = st1 == 200 and st2 == 200 and (c2 or 0) > 0
        return ok, {"turn1_status": st1, "turn1_cached": c1, "turn2_status": st2, "turn2_cached": c2,
                    "turn2_prompt_tokens": (d2.get("usage") or {}).get("prompt_tokens")}

    def _run_needle(self, target_tokens, depth, seed_offset):
        if self.served_window is None:
            self.gate_load()
        window = self.served_window or self.context
        if target_tokens > window and not self.allow_rescale:
            return False, {"skipped_or_failed": f"target {target_tokens} > served window {window}"}
        fit_target = min(target_tokens, window - 300)

        probe = records(400, seed=19)
        st_p, dp = self.chat("".join(probe) + "Reply OK.", thinking=False, max_tokens=1, temperature=0, timeout=600)
        if st_p != 200:
            return False, {"stage": "calibration", "status": st_p, "body": dp}
        per_line = (dp["usage"]["prompt_tokens"] - 10) / len(probe)

        n = max(1, int(fit_target / per_line))
        archive = CITIES[seed_offset % len(CITIES)]
        code = f"{7319 + 97 * seed_offset}-{WORDS[seed_offset % len(WORDS)]}-{42 + seed_offset}"
        lines = records(n, seed=100 + seed_offset)
        lines.insert(int(n * depth), f"The vault code for the {archive} archive is {code}.\n")
        q = f"\nIn the records above, one line states a vault code. What is the vault code for the {archive} archive? Reply with the code only."
        t0 = time.monotonic()
        st, d = self.chat("".join(lines) + q, thinking=False, max_tokens=48, temperature=0, timeout=7200)
        wall = round(time.monotonic() - t0, 2)
        ans = (((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "") if st == 200 else ""
        ok = st == 200 and code in ans
        return ok, {
            "target": target_tokens,
            "depth": depth,
            "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens") if st == 200 else None,
            "expected": code,
            "answer": ans,
            "wall_s": wall,
            "status": st,
        }

    def gate_needle_256k(self):
        ok1, r1 = self._run_needle(256000, 0.50, 1)
        ok2, r2 = self._run_needle(256000, 0.92, 2)
        return ok1 and ok2, {"depth_50": r1, "depth_92": r2}

    def gate_needle_1m(self):
        ok1, r1 = self._run_needle(1024000, 0.50, 3)
        ok2, r2 = self._run_needle(1024000, 0.92, 4)
        return ok1 and ok2, {"depth_50": r1, "depth_92": r2}

    def gate_soak(self):
        levels = [4, 8] if self.parallel >= 8 else [self.parallel]
        levels = sorted(set(levels))
        prefix = "".join(records(1500, seed=55))
        summary = {}
        all_ok = True
        for c in levels:
            out = [None] * c

            def worker(i):
                t0 = time.monotonic()
                try:
                    st, d = self.chat(
                        f"Stream-{i}-{time.time_ns()}\n" + prefix + "\n" + PROSE[i % len(PROSE)],
                        thinking=False, max_tokens=256, timeout=1800,
                    )
                    txt = (((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "") if st == 200 else ""
                    out[i] = {"status": st, "chars": len(txt), "loop": looping(txt), "wall_s": round(time.monotonic() - t0, 2)}
                except Exception as e:
                    out[i] = {"status": None, "error": repr(e)}

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            ok = all(r and r.get("status") == 200 and r.get("chars", 0) > 0 and not r.get("loop") for r in out)
            summary[f"c{c}"] = {"ok": ok, "streams": out}
            all_ok = all_ok and ok
        return all_ok, summary

    ALL_GATES = [
        "load", "chat", "reasoning", "tools", "structured", "vision",
        "drafted_exactness", "multilingual", "prefix_cache",
        "needle_256k", "needle_1m", "soak",
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--plain-url", help="optional un-drafted server URL for drafted==plain exactness comparison")
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--context", type=int, default=int(recipe.setting("CONTEXT", "1048576")))
    ap.add_argument("--parallel", type=int, default=int(recipe.setting("PARALLEL", "8")))
    ap.add_argument("--only", help="comma-separated subset of gates to run")
    ap.add_argument("--skip", help="comma-separated gates to skip")
    ap.add_argument("--allow-rescale", action="store_true", help="rescale needle target if served window < target")
    ap.add_argument("--out", help="write JSON results here")
    a = ap.parse_args(argv)

    g = Gates(a.url, a.model, a.context, a.parallel, plain_url=a.plain_url, allow_rescale=a.allow_rescale)
    selected = list(g.ALL_GATES)
    if a.only:
        only_set = [x.strip() for x in a.only.split(",") if x.strip()]
        selected = [x for x in selected if x in only_set]
    if a.skip:
        skip_set = {x.strip() for x in a.skip.split(",") if x.strip()}
        selected = [x for x in selected if x not in skip_set]

    results = {}
    for name in selected:
        t0 = time.monotonic()
        try:
            ok, detail = getattr(g, f"gate_{name}")()
        except Exception as e:
            ok, detail = False, {"exception": repr(e)}
        dt = round(time.monotonic() - t0, 2)
        results[name] = {"ok": ok, "wall_s": dt, "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name:<18} ({dt}s)  {json.dumps(detail)[:240]}", flush=True)

    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")

    failed = [k for k, v in results.items() if not v["ok"]]
    print(f"gates: {len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
