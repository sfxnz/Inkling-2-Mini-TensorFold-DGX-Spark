#!/usr/bin/env bash
# Checks tf-inkling2 on this host (no model, no serve). Writes artifacts/verify-image/ by default.
set -euo pipefail
DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"
OUT=${1:-$DIR/artifacts/verify-image}
mkdir -p "$OUT"
run() { docker run --rm --gpus all "$IMAGE" "$@"; }

run tensorfold-native --version | tee "$OUT/version.txt"
grep -qF "$TF_VERSION" "$OUT/version.txt" || { echo "FAIL: --version is not $TF_VERSION" >&2; exit 1; }

run tensorfold-native capabilities --json > "$OUT/capabilities.json"
jq -e '.chip == "nvidia-sm121"' "$OUT/capabilities.json" >/dev/null || { echo "FAIL: chip is not nvidia-sm121" >&2; exit 1; }
for lib in cuBLASLt "TensorFold grammar" libjpeg-turbo libwebp; do
  jq -e --arg n "$lib" '.optional_libraries[] | select(.name == $n) | .found == true and .usable != false' \
    "$OUT/capabilities.json" >/dev/null || { echo "FAIL: $lib not found or not usable" >&2; exit 1; }
done
jq -c '.chip, .version, (.optional_libraries[] | {name, found, usable, problem})' "$OUT/capabilities.json" | tee "$OUT/capabilities-summary.txt"

run sh -c 'dpkg-query -W libcublas-13-0;
  ls -l /opt/tensorfold/lib;
  for f in /opt/tensorfold/lib/*.so*; do echo "== $f"; ldd "$f" | grep -E "not found|=>"; done;
  grep -ao "libjpeg-turbo version [^ ]*" /opt/tensorfold/lib/libturbojpeg.so.0 | head -1' | tee "$OUT/libraries.txt"

docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}} size={{.Size}}' | tee "$OUT/labels.txt"
echo OK
