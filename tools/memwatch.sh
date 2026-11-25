#!/usr/bin/env bash
# Unified-memory watchdog for 1x DGX Spark.
# Samples GPU residency (nvidia-smi / drm-resident), container anon RAM, system swap-out, host CPU load, and MemAvailable.
# Usage:
#   tools/memwatch.sh OUT [--interval S] [--floor-gib G] [--container NAME] [--enforce]
#   tools/memwatch.sh OUT [SECONDS]
set -euo pipefail

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$DIR/recipe.conf" ]]; then
  # shellcheck disable=SC1091
  source "$DIR/recipe.conf"
fi

OUT=${1:?usage: tools/memwatch.sh OUT [--interval S] [--floor-gib G] [--container NAME] [--enforce]}
shift

INTERVAL=5
FLOOR_GIB=${MEM_FLOOR_GIB:-8.0}
CONTAINER=${CONTAINER_NAME:-tf-inkling2}
ENFORCE=0

if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
  INTERVAL=$1
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL=${2:?missing --interval}; shift 2 ;;
    --floor-gib) FLOOR_GIB=${2:?missing --floor-gib}; shift 2 ;;
    --container) CONTAINER=${2:?missing --container}; shift 2 ;;
    --enforce) ENFORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$OUT")"

sample_container() {
  local c=$1 pids="" cid cg="" anon_bytes=0 gpu_mib=0 p m
  if ! docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -qx true; then
    echo "0.00 0.00"
    return
  fi
  cid=$(docker inspect -f '{{.Id}}' "$c" 2>/dev/null || true)
  if [[ -n $cid ]]; then
    for d in "/sys/fs/cgroup/system.slice/docker-$cid.scope" "/sys/fs/cgroup/docker/$cid"; do
      if [[ -f "$d/memory.stat" ]]; then
        cg=$d
        break
      fi
    done
  fi
  if [[ -n $cg ]]; then
    anon_bytes=$(awk '/^anon /{print $2; exit}' "$cg/memory.stat" 2>/dev/null || echo 0)
  fi
  pids=$(docker top "$c" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ' || true)
  if command -v nvidia-smi >/dev/null 2>&1 && [[ -n $pids ]]; then
    local smi_out
    smi_out=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    for p in $pids; do
      m=$(awk -F', ' -v pid="$p" '$1 == pid {s += $2} END {print s+0}' <<<"$smi_out")
      gpu_mib=$((gpu_mib + m))
    done
  fi
  if [[ $anon_bytes -eq 0 && -n $pids ]]; then
    local kb=0
    for p in $pids; do
      if [[ -r "/proc/$p/smaps_rollup" ]]; then
        kb=$((kb + $(awk '/^Anonymous:/{print $2; exit}' "/proc/$p/smaps_rollup" 2>/dev/null || echo 0)))
      fi
    done
    anon_bytes=$((kb * 1024))
  fi
  awk -v g="$gpu_mib" -v a="$anon_bytes" 'BEGIN {printf "%.2f %.2f", g / 1024.0, a / 1073741824.0}'
}

prev_pswpout=$(awk '/^pswpout /{print $2; exit}' /proc/vmstat 2>/dev/null || echo 0)
read -r _ u1 n1 s1 i1 w1 q1 sq1 _ </proc/stat
prev_busy=$((u1 + n1 + s1 + q1 + sq1))
prev_tot=$((prev_busy + i1 + w1))

echo "ts_utc mem_avail_gib gpu_res_gib proc_anon_gib swap_used_gib pswpout_delta load_1m host_cpu_pct alert" >> "$OUT"

while :; do
  sleep "$INTERVAL"
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  mem_avail_gib=$(awk '/^MemAvailable:/ {printf "%.2f", $2 / 1048576}' /proc/meminfo)
  swap_used_gib=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END {printf "%.2f", (t-f)/1048576}' /proc/meminfo)
  cur_pswpout=$(awk '/^pswpout /{print $2; exit}' /proc/vmstat 2>/dev/null || echo 0)
  pswp_delta=$((cur_pswpout - prev_pswpout))
  prev_pswpout=$cur_pswpout

  load_1m=$(awk '{print $1}' /proc/loadavg)
  read -r _ u2 n2 s2 i2 w2 q2 sq2 _ </proc/stat
  cur_busy=$((u2 + n2 + s2 + q2 + sq2))
  cur_tot=$((cur_busy + i2 + w2))
  dt=$((cur_tot - prev_tot))
  db=$((cur_busy - prev_busy))
  prev_busy=$cur_busy
  prev_tot=$cur_tot
  cpu_pct=$(awk -v b="$db" -v t="$dt" 'BEGIN {if (t > 0) printf "%.1f", 100.0 * b / t; else print "0.0"}')

  read -r gpu_res_gib proc_anon_gib < <(sample_container "$CONTAINER")

  alert="ok"
  below=$(awk -v a="$mem_avail_gib" -v f="$FLOOR_GIB" 'BEGIN {print (a < f) ? 1 : 0}')
  if [[ $below -eq 1 ]]; then
    alert="LOW_MEM"
  elif [[ $pswp_delta -gt 1024 ]]; then
    alert="SWAP_OUT"
  fi

  echo "$ts $mem_avail_gib $gpu_res_gib $proc_anon_gib $swap_used_gib $pswp_delta $load_1m $cpu_pct $alert" >> "$OUT"

  if [[ $ENFORCE -eq 1 && $below -eq 1 ]]; then
    echo "memwatch: MemAvailable ${mem_avail_gib} GiB < floor ${FLOOR_GIB} GiB; triggering graceful stop" >&2
    if [[ -x "$DIR/stop.sh" ]]; then
      "$DIR/stop.sh" || true
    fi
    exit 2
  fi
done
