#!/usr/bin/env bash
# Reproduce the patched TensorFold image (tf-inkling2:m3-p9) from a clean checkout of BASE + engine-patches/ (0001..0023).
# Serializes Docker builds under /tmp/inference-hillclimb.lock.
set -euo pipefail

if [[ ${1:-} != --locked ]]; then
  exec flock /tmp/inference-hillclimb.lock "$0" --locked "$@"
fi
shift

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$DIR/recipe.conf"

TAG=${1:-$IMAGE}
BUILD_GRAMMAR=${BUILD_GRAMMAR:-0}
BUILDER_IMG="tf-inkling2-builder:tmp-$$"
EXTRACT_NAME="tf-inkling2-extract-$$"
OUT_DIR=$(mktemp -d)
cleanup() {
  docker rm -f "$EXTRACT_NAME" >/dev/null 2>&1 || true
  docker rmi -f "$BUILDER_IMG" >/dev/null 2>&1 || true
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

PATCH_SHA=$(cd "$DIR/engine-patches" && while IFS= read -r p; do [[ -n $p && $p != \#* ]] && sha256sum "$p"; done < series | sha256sum | awk '{print $1}')

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "... building base runtime image $BASE_IMAGE"
  docker build -t "$BASE_IMAGE" --build-arg "TF_VERSION=$TF_VERSION" --build-arg "TF_SHA256=$TF_SHA256" "$DIR/docker"
fi

echo "... building patched TensorFold binary in Docker (BUILD_GRAMMAR=$BUILD_GRAMMAR)"
docker build -f "$DIR/docker/Dockerfile.builder" --build-arg "BUILD_GRAMMAR=$BUILD_GRAMMAR" -t "$BUILDER_IMG" "$DIR"

docker create --name "$EXTRACT_NAME" "$BUILDER_IMG" >/dev/null
docker cp "$EXTRACT_NAME:/out/tensorfold-native" "$OUT_DIR/tensorfold-native"
BIN_SHA=$(sha256sum "$OUT_DIR/tensorfold-native" | awk '{print $1}')

GRAMMAR_ARGS=()
if [[ $BUILD_GRAMMAR == 1 ]]; then
  docker cp "$EXTRACT_NAME:/out/libtfgrammar.so" "$OUT_DIR/libtfgrammar.so"
  GRAMMAR_SHA=$(sha256sum "$OUT_DIR/libtfgrammar.so" | awk '{print $1}')
  GRAMMAR_ARGS=(--build-arg "GRAMMAR_SHA256=$GRAMMAR_SHA")
fi

echo "... overlaying patched binary onto $BASE_IMAGE -> $TAG"
docker build -f "$DIR/docker/Dockerfile.patched" \
  --build-context "engine=$OUT_DIR" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BIN_SHA256=$BIN_SHA" \
  "${GRAMMAR_ARGS[@]}" \
  --build-arg "PATCHES=$PATCH_SHA" \
  -t "$TAG" "$DIR/docker"

echo "built $TAG (patches=$PATCH_SHA)"
