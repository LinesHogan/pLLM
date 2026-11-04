#!/bin/bash
# Verify the author loop, GPU core and changed-path feature groups.
set -uo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1
export PY=${PY:-/workspace/pllm/.venv/bin/python}
export PATH="$(dirname "$PY"):$PATH" PYTHONPATH=$ROOT HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
options=(); requested=(); listing=''; explain=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      cat <<'HELP'
Usage: ROOT=<checkout> PY=<venv/bin/python> bash tools/full_verify.sh [options] [cpu|gpu|stage...]
Default: static checks + tier 0 CPU + tier 1 GPU core + affected tier 2 groups.
--base REF  Compare changed paths with merge-base(REF, HEAD); default main, including dirty paths.
--all       Run every feature group for scheduled/release sweeps (TP needs two GPUs and vLLM 0.32).
--explain   Print selected feature groups and the changed paths requiring them; do not run tests.
--list cpu|gpu|all  Print the selected stages, suitable for cluster_verify.
Stages: ruff format pyright typing layers docs cpu gpu-core graph-replay enginecore-capture enginecore-io,
        plus feature group names from tools/verify_groups.py. Explicit stages run only those stages.
CPU tests hide CUDA. Local GPU work holds /tmp/pllm-gpu.lock, affinity 0-15; Slurm owns remote affinity.
Logs, exit codes and per-stage wall seconds go to LOGS (default docs/cleanup/verify). No test retries.
HELP
      exit 0 ;;
    --all) options+=(--all); shift ;;
    --base) options+=(--base "$2"); shift 2 ;;
    --explain) explain=yes; shift ;;
    --list) listing=$2; shift 2 ;;
    *) requested+=("$1"); shift ;;
  esac
done
[ "$explain" = no ] || exec "$PY" tools/verify_groups.py "${options[@]}" --explain
selection=$("$PY" tools/verify_groups.py "${options[@]}" --explain) || exit 1
features=(); while IFS=$'\t' read -r group reason; do [ -z "$group" ] || features+=("$group"); done <<< "$selection"
CPU_STAGES=(ruff format pyright typing layers docs cpu)
GPU_STAGES=(gpu-core graph-replay enginecore-capture enginecore-io)
for group in "${features[@]}"; do
  if [ "$group" = release ]; then CPU_STAGES+=(release); else GPU_STAGES+=("$group"); fi
done
if [ -n "$listing" ]; then
  case "$listing" in
    cpu) printf '%s\n' "${CPU_STAGES[@]}" ;;
    gpu) printf '%s\n' "${GPU_STAGES[@]}" ;;
    all) printf '%s\n' "${CPU_STAGES[@]}" "${GPU_STAGES[@]}" ;;
    *) exit 2 ;;
  esac
  exit 0
fi
stages=("${CPU_STAGES[@]}" "${GPU_STAGES[@]}")
if [ "${#requested[@]}" -gt 0 ]; then
  case "${requested[*]}" in
    cpu) stages=("${CPU_STAGES[@]}") ;;
    gpu) stages=("${GPU_STAGES[@]}") ;;
    *) stages=("${requested[@]}") ;;
  esac
fi
LOGS=${LOGS:-$ROOT/docs/cleanup/verify}; mkdir -p "$LOGS"
printf '%s' "$selection" > "$LOGS/selection.tsv"
[ -z "$selection" ] || printf 'Selected feature groups (group / changed path):\n%s\n' "$selection"
if git rev-parse --git-dir >/dev/null 2>&1; then
  commit=$(git rev-parse HEAD); dirty=no
  [ -z "$(git status --porcelain)" ] || dirty=yes
  printf 'commit=%s dirty=%s\n' "$commit" "$dirty" > "$LOGS/verification.txt"
else
  head -1 "$ROOT/verification.txt" > "$LOGS/verification.txt"
fi
PROJECT=()
if [ -f pyrightconfig.vllm032.json ] && "$PY" -c "import importlib.metadata as m; assert m.version('vllm').startswith('0.32')" 2>/dev/null; then PROJECT=(-p pyrightconfig.vllm032.json); fi
CPU=(); GPU=()
if [ -z "${SLURM_JOB_ID:-}" ]; then CPU=(taskset -c 0-15);  fi
pytest_cpu() { env CUDA_VISIBLE_DEVICES='' "${CPU[@]}" "$PY" -m pytest -q -p no:cacheprovider --durations=0 "$@"; }
pytest_gpu() { "${GPU[@]}" "${CPU[@]}" "$PY" -m pytest -q -p no:cacheprovider --durations=0 "$@"; }
smoke() { "${GPU[@]}" "${CPU[@]}" "$PY" "test/gpu/$1.py" "${@:2}"; }
run_stage() {
  local rc=0
  case "$1" in
    ruff) "$PY" -m ruff check ;;
    format) "$PY" -m ruff format --check ;;
    pyright) "$PY" -m pyright --pythonpath "$PY" "${PROJECT[@]}" ;;
    typing) "$PY" tools/check_typing.py ;;
    layers) "$PY" tools/check_import_layers.py ;;
    docs) "$PY" tools/check_docs.py ;;
    cpu) pytest_cpu -m 'not gpu' test/boundary test/workloads ;;
    release) pytest_cpu test/release ;;
    gpu-core) pytest_gpu --import-mode=importlib -m gpu test/boundary test/workloads test/gpu/core ;;
    graph-replay) smoke esamp_cuda_graph_replay_smoke ;;
    enginecore-capture) smoke enginecore_smoke --captured-only ;;
    enginecore-io) smoke enginecore_smoke --gpu-io-only ;;
    adaptive) pytest_gpu test/gpu/features/test_adaptive_depth.py ;;
    residual) pytest_gpu test/gpu/features/test_custom_residual.py test/gpu/features/test_custom_kernels.py test/gpu/features/test_engram_lite.py ;;
    producers) pytest_gpu test/gpu/features/test_state_producers.py ;;
    routes)
      pytest_gpu test/gpu/features/test_composed_model_sites.py || rc=1
      for name in vllm_layer_loop_smoke vllm_recirculation_smoke vllm_route_updates_smoke; do smoke "$name" || rc=1; done
      return "$rc" ;;
    positions) smoke vllm_step_loop_smoke ;;
    feeds) pytest_gpu test/gpu/features/test_state_feed_cuda.py test/gpu/features/test_video_session.py ;;
    esamp) smoke vllm_esamp_smoke ;;
    feedback) smoke vllm_override_smoke || rc=1; smoke fttt_cuda_smoke || rc=1; smoke vllm_grpo_smoke || rc=1; return "$rc" ;;
    reft) smoke vllm_reft_smoke ;;
    inplace) smoke vllm_inplace_ttt_smoke ;;
    tlm) smoke vllm_tlm_smoke ;;
    enginecore) smoke enginecore_smoke || rc=1; smoke vllm_mrv1_smoke || rc=1; return "$rc" ;;
    tp) pytest_gpu test/gpu/vllm_tp_smoke.py ;;
    *) echo "Unknown stage: $1; use --help" >&2; return 2 ;;
  esac
}
gpu_lock_held() {
  [ -z "${PLLM_GPU_LEASE:-}" ] || return 0
  local descriptor
  # An outer flock passes its open file description; reopening the file would deadlock.
  for descriptor in /proc/$$/fd/*; do
    if [ "$(readlink "$descriptor")" = /tmp/pllm-gpu.lock ] && flock -n "${descriptor##*/}"; then return 0; fi
  done
  return 1
}
failed=0
for stage in "${stages[@]}"; do
  gpu_lock=''
  if [ -z "${SLURM_JOB_ID:-}" ] && ! gpu_lock_held; then
    case "$stage" in
      ruff|format|pyright|typing|layers|docs|cpu|release) ;;
      *) exec {gpu_lock}>/tmp/pllm-gpu.lock; flock "$gpu_lock" ;;
    esac
  fi
  start=$(date +%s%N)
  run_stage "$stage" > "$LOGS/$stage.log" 2>&1; rc=$?
  end=$(date +%s%N)
  awk -v start="$start" -v end="$end" 'BEGIN {printf "%.3f\n", (end-start)/1e9}' > "$LOGS/$stage.seconds"
  printf '%s\n' "$rc" > "$LOGS/$stage.exit"
  printf '%s exit=%s wall=%ss\n' "$stage" "$rc" "$(cat "$LOGS/$stage.seconds")"
  tail -1 "$LOGS/$stage.log"
  [ -z "$gpu_lock" ] || { flock -u "$gpu_lock"; exec {gpu_lock}>&-; }
  [ "$rc" -eq 0 ] || failed=1
done
if [ -n "${commit:-}" ] && [ "$dirty" = no ]; then
  [ "$(git rev-parse HEAD)" = "$commit" ] && [ -z "$(git status --porcelain)" ] || failed=1
fi
exit "$failed"
