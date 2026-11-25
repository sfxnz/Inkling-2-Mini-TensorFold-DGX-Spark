#!/usr/bin/env bash
# Launch Inkling-2-Mini on a single NVIDIA DGX Spark (1x GB10) with TensorFold.
# Settings live in recipe.conf; environment variables override any default.
# VALIDATE_ONLY=1 runs all preflight checks read-only, prints the docker command, and exits 1 if run.sh would refuse.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"

VALIDATE_ONLY=${VALIDATE_ONLY:-0}
REFUSED=0
ALREADY_STARTING=0
GIB=1073741824

die() { echo "error: $*" >&2; exit 1; }
ok() { echo "ok     $*"; }
quote() { local q; printf -v q '%q ' "$@"; echo "${q% }"; }
refuse() {
  [[ $VALIDATE_ONLY == 1 ]] || die "$*"
  echo "REFUSE $*"
  REFUSED=$((REFUSED + 1))
}
act() {
  local what=$1; shift
  if [[ $VALIDATE_ONLY == 1 ]]; then
    echo "WOULD  $what: $(quote "$@")"
  else
    echo "...    $what"
    "$@"
  fi
}

probe_tools() {
  local t
  for t in "$@"; do
    command -v "$t" >/dev/null || echo "$t"
  done
}

# List running GPU containers that do not carry LABEL.
probe_gpu_containers() {
  local ids id name image label gpu
  ids=$(docker ps -q) || return 1
  for id in $ids; do
    IFS='|' read -r name image label gpu < <(docker inspect --format \
      '{{.Name}}|{{.Config.Image}}|{{index .Config.Labels "ai-lab.recipe"}}|{{.HostConfig.Runtime}} {{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' "$id")
    [[ "ai-lab.recipe=$label" == "$1" ]] && continue
    if [[ $gpu == nvidia* || $gpu == *'"gpu"'* || $gpu == */dev/nvidia* ]]; then
      echo "${name#/} $image"
    fi
  done
}

probe_keyfile() {
  if [[ ! -f $1 ]]; then
    echo missing
  elif [[ $(stat -c %a "$1") != 600 ]]; then
    echo "mode $(stat -c %a "$1")"
  else
    echo ok
  fi
}

probe_ours() { docker ps -a --filter "label=$1" --format '{{.Names}} {{.State}}'; }
probe_image() { docker image inspect --format '{{.Id}} {{index .Config.Labels "tensorfold.archive.sha256"}}' "$1" 2>/dev/null || true; }
probe_mem() { free -b | awk '/^Mem:/ {print $7}'; }
probe_ports() { ss -Htln | awk '{n = split($4, a, ":"); print a[n]}' | grep -Fx -f <(printf '%s\n' "$@") | sort -u || true; }

snapshot_check() {
  local d=$1 f shards n total mtp_size
  for f in config.json tokenizer.json chat_template.jinja processor_config.json model.safetensors.index.json \
    model-00001-of-00001.safetensors mtp.safetensors; do
    [[ -s $d/$f ]] || { echo "missing $f in $d"; return 1; }
  done
  shards=$(grep -o '"[^"]*\.safetensors"' "$d/model.safetensors.index.json" | tr -d '"' | sort -u)
  n=$(grep -c . <<<"$shards" || true)
  [[ $n == "$2" ]] || { echo "index names $n shards, want $2"; return 1; }
  for f in $shards; do
    [[ -f $d/$f ]] || { echo "missing shard $f in $d"; return 1; }
  done
  # shellcheck disable=SC2086
  total=$(cd "$d" && stat -Lc %s $shards | awk '{s += $1} END {printf "%d", s}')
  [[ $total == "$3" ]] || { echo "model shards hold $total bytes, want $3"; return 1; }
  mtp_size=$(stat -Lc %s "$d/mtp.safetensors")
  [[ $mtp_size == "$MTP_BYTES" ]] || { echo "mtp.safetensors holds $mtp_size bytes, want $MTP_BYTES"; return 1; }
  echo complete
}

drafter_check() {
  local d=$1 size
  for f in config.json model.safetensors; do
    [[ -s $d/$f ]] || { echo "missing $f in $d"; return 1; }
  done
  size=$(stat -Lc %s "$d/model.safetensors")
  [[ $size == "$DRAFTER_SHARD_BYTES" ]] || { echo "drafter model.safetensors holds $size bytes, want $DRAFTER_SHARD_BYTES"; return 1; }
  echo complete
}

container_argv() {
  local cache=() keys=() mount=() env=() extra=() vision_args=() e
  [[ -n $PROMPT_CACHE_GIB ]] && cache=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
  [[ -n $ENGINE_CACHE ]] && mount+=(-v "$ENGINE_CACHE:/root/.cache/tensorfold")
  for e in $EXTRA_ENV; do
    env+=(-e "$e")
  done
  if [[ -n $EXTRA_ARGS ]]; then
    read -ra extra <<<"$EXTRA_ARGS"
  fi
  if [[ $VISION == 1 ]]; then
    vision_args=(--vision --vision-video-tokens "$VIDEO_TOKENS")
  fi
  if [[ -n $DRAFTER ]]; then
    mount+=(-v "$DRAFTER_REPO_DIR:$DRAFTER_REPO_DIR:ro")
    extra+=(--drafter "$DRAFTER_SNAPSHOT" --drafter-bits "$DRAFTER_BITS")
  fi
  if [[ -n $API_KEY_FILE ]]; then
    keys=(--api-key-file "$API_KEY_FILE")
    mount+=(-v "$API_KEY_FILE:$API_KEY_FILE:ro")
  fi
  printf '%s\n' docker run -d --name "$CONTAINER_NAME" --label "$LABEL" \
    --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit core=0 --oom-score-adj 1000 \
    -v "$REPO_DIR:$REPO_DIR:ro" "${mount[@]}" \
    -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB" "${env[@]}" \
    "$IMAGE" tensorfold-native serve "$SNAPSHOT" --host 0.0.0.0 --port "$PORT" --name "$SERVED_NAME" \
    --context "$CONTEXT" --parallel "$PARALLEL" "${vision_args[@]}" \
    "${cache[@]}" "${keys[@]}" "${extra[@]}" --no-update-check
}

logs_tail() {
  echo "--- $CONTAINER_NAME ($(hostname -s)), last 40 lines"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 || true
}

endpoint() {
  echo "endpoint  http://$(hostname -s):$PORT/v1"
  echo "try       curl -s http://$(hostname -s):$PORT/v1/chat/completions -H 'content-type: application/json' \\"
  echo "            -d '{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
  echo "stop      $DIR/stop.sh"
}

running() {
  [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null) == true ]]
}

check_config() {
  local v
  [[ $VALIDATE_ONLY =~ ^[01]$ ]] || die "VALIDATE_ONLY=$VALIDATE_ONLY must be 0 or 1"
  [[ $FORCE =~ ^[01]$ ]] || die "FORCE=$FORCE must be 0 or 1"
  [[ $VISION =~ ^[01]$ ]] || die "VISION=$VISION must be 0 or 1"
  for v in PORT CONTEXT PARALLEL MEM_GATE_GIB READY_TIMEOUT; do
    [[ ${!v} =~ ^[1-9][0-9]*$ ]] || die "$v=${!v} is not a positive integer"
  done
  [[ $DRAFTER_BITS =~ ^(0|4)$ ]] || die "DRAFTER_BITS=$DRAFTER_BITS must be 0 or 4"
  [[ $MEMORY_RESERVE_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB is not a number of GiB"
  [[ $MEM_FLOOR_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEM_FLOOR_GIB=$MEM_FLOOR_GIB is not a number of GiB"
  if [[ ! $VIDEO_TOKENS =~ ^[1-9][0-9]*$ ]] || (( VIDEO_TOKENS < 64 || VIDEO_TOKENS > 262144 )); then
    die "VIDEO_TOKENS=$VIDEO_TOKENS must be 64 to 262144"
  fi
  [[ -z $PROMPT_CACHE_GIB || $PROMPT_CACHE_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] ||
    die "PROMPT_CACHE_GIB=$PROMPT_CACHE_GIB is not a number of GiB"
  [[ -z $API_KEY_FILE || ($API_KEY_FILE == /* && $API_KEY_FILE != *[[:space:]]*) ]] ||
    die "API_KEY_FILE=$API_KEY_FILE must be an absolute path without spaces"
  [[ $HF_CACHE == /* && $HF_CACHE != *[[:space:]]* ]] || die "HF_CACHE=$HF_CACHE must be an absolute path without spaces"
  [[ -z $ENGINE_CACHE || $ENGINE_CACHE =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    die "ENGINE_CACHE=$ENGINE_CACHE must be a docker volume name"
  for v in $EXTRA_ENV; do
    [[ $v =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] || die "EXTRA_ENV word '$v' is not VAR=VALUE"
  done
  REPO_DIR=$HF_CACHE/hub/models--${MODEL//\//--}
  SNAPSHOT=$REPO_DIR/snapshots/$REVISION
  if [[ -n $DRAFTER ]]; then
    DRAFTER_REPO_DIR=$HF_CACHE/hub/models--${DRAFTER//\//--}
    DRAFTER_SNAPSHOT=$DRAFTER_REPO_DIR/snapshots/$DRAFTER_REVISION
  fi
}

check_pin() {
  if [[ $REVISION =~ ^[0-9a-f]{40}$ && $SHARDS =~ ^[1-9][0-9]*$ && $SHARD_BYTES =~ ^[1-9][0-9]*$ ]]; then
    PINNED=1
  else
    PINNED=0
    refuse "target model pack is not pinned: REVISION=$REVISION SHARDS=$SHARDS SHARD_BYTES=$SHARD_BYTES"
  fi
  if [[ -n $DRAFTER && ! ($DRAFTER_REVISION =~ ^[0-9a-f]{40}$ && $DRAFTER_SHARD_BYTES =~ ^[1-9][0-9]*$) ]]; then
    PINNED=0
    refuse "drafter pack is not pinned: DRAFTER_REVISION=$DRAFTER_REVISION DRAFTER_SHARD_BYTES=$DRAFTER_SHARD_BYTES"
  fi
}

print_config() {
  local v
  echo "config (recipe.conf, environment wins):"
  while IFS= read -r v; do
    echo "  $v=${!v}"
  done < <(sed -n 's/^\([A-Z][A-Z0-9_]*\)=.*/\1/p' "$DIR/recipe.conf")
  echo "  snapshot=$SNAPSHOT"
  [[ -n $DRAFTER ]] && echo "  drafter_snapshot=$DRAFTER_SNAPSHOT"
}

step_tools() {
  local missing
  missing=$(probe_tools docker curl hf ss free stat awk grep)
  [[ -z $missing ]] || die "host lacks: $(echo "$missing" | xargs)"
  ok "host $(hostname -s) has docker curl hf ss free stat awk grep"
}

step_already_up() {
  local ours
  ours=$(probe_ours "$LABEL")
  if ! grep -Fxq "$CONTAINER_NAME running" <<<"$ours"; then
    return 0
  fi
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "already serving: $CONTAINER_NAME on $(hostname -s) answers /health"
    endpoint
    exit 0
  fi
  ALREADY_STARTING=1
  echo "$CONTAINER_NAME runs but /health does not answer yet: waiting for it instead of restarting"
}

step_preflight() {
  local gpu
  if [[ -n $API_KEY_FILE ]]; then
    case $(probe_keyfile "$API_KEY_FILE") in
      ok) ok "API_KEY_FILE $API_KEY_FILE is mode 600" ;;
      missing) refuse "API_KEY_FILE $API_KEY_FILE is missing (install -m 600)" ;;
      *) refuse "API_KEY_FILE $API_KEY_FILE must be mode 600" ;;
    esac
  fi
  if ! gpu=$(probe_gpu_containers "$LABEL"); then
    refuse "cannot list docker containers"
  elif [[ -n $gpu ]]; then
    while read -r g; do
      refuse "host runs GPU container $g; refusing to collide on the GPU"
    done <<<"$gpu"
  else
    ok "no other GPU container running"
  fi
}

step_image() {
  local h
  h=$(probe_image "$IMAGE")
  if [[ -n $h && ${h#* } == "$TF_SHA256" ]]; then
    ok "image $IMAGE: ${h%% *}, TensorFold archive sha256 ${TF_SHA256:0:12}"
  elif [[ $IMAGE == "$BASE_IMAGE" ]]; then
    [[ -n $h ]] && echo "note   $IMAGE carries archive sha256 ${h#* }, want $TF_SHA256: rebuilding"
    act "build $IMAGE from docker/ under flock" flock /tmp/inference-hillclimb.lock \
      docker build -t "$IMAGE" --build-arg "TF_VERSION=$TF_VERSION" --build-arg "TF_SHA256=$TF_SHA256" "$DIR/docker"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      h=$(probe_image "$IMAGE")
      [[ ${h#* } == "$TF_SHA256" ]] || die "built $IMAGE carries archive sha256 '${h#* }', want $TF_SHA256"
    fi
  else
    refuse "image $IMAGE not present locally (or base sha256 mismatch); build it via docker/Dockerfile.patched first"
  fi
}

step_weights() {
  local st inc=() p pats
  if [[ $PINNED == 0 ]]; then
    echo "skip   weights: pack not pinned"
    return
  fi
  if st=$(snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES"); then
    ok "target snapshot complete: $SHARDS model shard + mtp, $SHARD_BYTES + $MTP_BYTES bytes"
  else
    echo "note   target snapshot: $st"
    read -ra pats <<<"$HF_INCLUDE"
    for p in "${pats[@]}"; do
      inc+=(--include "$p")
    done
    act "download $MODEL at $REVISION" hf download "$MODEL" --revision "$REVISION" --cache-dir "$HF_CACHE/hub" "${inc[@]}"
    if [[ $VALIDATE_ONLY == 0 ]]; then
      st=$(snapshot_check "$SNAPSHOT" "$SHARDS" "$SHARD_BYTES") || die "after download, target snapshot: $st"
    fi
  fi

  if [[ -n $DRAFTER ]]; then
    if st=$(drafter_check "$DRAFTER_SNAPSHOT"); then
      ok "drafter snapshot complete: $DRAFTER_SHARD_BYTES bytes"
    else
      echo "note   drafter snapshot: $st"
      act "download $DRAFTER at $DRAFTER_REVISION" hf download "$DRAFTER" --revision "$DRAFTER_REVISION" --cache-dir "$HF_CACHE/hub"
      if [[ $VALIDATE_ONLY == 0 ]]; then
        st=$(drafter_check "$DRAFTER_SNAPSHOT") || die "after download, drafter snapshot: $st"
      fi
    fi
  fi
}

step_clear_ours() {
  local ours
  ours=$(probe_ours "$LABEL")
  if [[ -n $ours ]]; then
    echo "note   ours left from an earlier run: $(echo "$ours" | xargs)"
    act "stop and remove them" "$DIR/stop.sh"
    CLEARED_OURS=1
  else
    CLEARED_OURS=0
  fi
}

step_ports() {
  local busy
  if [[ $VALIDATE_ONLY == 1 && $CLEARED_OURS == 1 ]]; then
    echo "WOULD  check port $PORT once ours are gone"
    return
  fi
  busy=$(probe_ports "$PORT" | xargs)
  if [[ -n $busy ]]; then
    refuse "port $busy already listens on $(hostname -s) (check with ss -ltnp)"
  else
    ok "port $PORT free"
  fi
}

step_memory() {
  local avail avail_gib
  avail=$(probe_mem)
  avail_gib=$(awk -v b="$avail" 'BEGIN {printf "%.1f", b / 1073741824}')
  if (( avail < MEM_GATE_GIB * GIB )); then
    echo "--- free -h on $(hostname -s)"
    free -h
    if [[ $FORCE == 1 ]]; then
      echo "FORCE=1: starting below ${MEM_GATE_GIB} GiB gate (${avail_gib} GiB)"
    else
      refuse "MemAvailable ${avail_gib} GiB below ${MEM_GATE_GIB} GiB gate (free memory or set FORCE=1)"
    fi
  else
    ok "MemAvailable ${avail_gib} GiB >= ${MEM_GATE_GIB} GiB"
  fi
}

step_start() {
  local argv
  mapfile -t argv < <(container_argv)
  if [[ $VALIDATE_ONLY == 1 ]]; then
    echo "argv   $(hostname -s): $(quote "${argv[@]}")"
    return
  fi
  echo "...    start $CONTAINER_NAME on $(hostname -s)"
  "${argv[@]}" >/dev/null || { logs_tail; "$DIR/stop.sh"; die "$CONTAINER_NAME failed to start"; }
}

step_wait() {
  local t0=$SECONDS next=30
  if [[ $VALIDATE_ONLY == 1 ]]; then
    echo "WOULD  wait up to ${READY_TIMEOUT}s for http://127.0.0.1:$PORT/health"
    return
  fi
  echo "...    wait for /health (up to ${READY_TIMEOUT}s)"
  until curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    local why=""
    running || why="$CONTAINER_NAME exited"
    (( SECONDS - t0 < READY_TIMEOUT )) || why="no /health after ${READY_TIMEOUT}s"
    if [[ -n $why ]]; then
      echo "error: $why" >&2
      logs_tail
      "$DIR/stop.sh"
      exit 1
    fi
    if (( SECONDS - t0 >= next )); then
      echo "       $((SECONDS - t0))s"
      next=$((next + 30))
    fi
    sleep 2
  done
  echo "ready after $((SECONDS - t0))s"
  docker logs "$CONTAINER_NAME" 2>&1 | grep -E '^(\[tensorfold\] )?(warning|CUDA|serving|inkling|draft)' || true
  endpoint
}

check_config
echo "== $MODEL, TensorFold $TF_VERSION, 1x DGX Spark ($(hostname -s))"
[[ $VALIDATE_ONLY == 0 ]] || print_config
check_pin
step_tools
step_already_up
if [[ $ALREADY_STARTING == 0 ]]; then
  step_preflight
  step_image
  step_weights
  step_clear_ours
  step_ports
  step_memory
  step_start
fi
step_wait
if [[ $VALIDATE_ONLY == 1 ]]; then
  if (( REFUSED > 0 )); then
    echo "VALIDATE_ONLY: run.sh would refuse ($REFUSED reasons above)"
    exit 1
  fi
  echo "VALIDATE_ONLY: every check passes; run.sh would start $CONTAINER_NAME"
fi
