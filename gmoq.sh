#!/bin/bash
#PBS -N vllm_omni_sweep
#PBS -l select=1:ncpus=24:ngpus=3:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -P personal-es0001an

module load singularity
set -euo pipefail
cd "$PBS_O_WORKDIR"

# ── Raise open-file limit early ───────────────────────────────────────────────
ulimit -n 65535 2>/dev/null \
  || ulimit -n "$(ulimit -Hn)" 2>/dev/null \
  || true
echo "ulimit -n = $(ulimit -n)"

# --- Directory Setup ---
SCRATCH_DIR=/scratch/users/ntu/es0001an
IMG=$SCRATCH_DIR/vllm-omni-try/vllm-omni-official.sif
HF_CACHE=$SCRATCH_DIR/hf_cache
VLLM_WORKDIR=$SCRATCH_DIR/vllm-omni-try
DATASET_DIR=$SCRATCH_DIR/dataset_generated
PYSITE=$SCRATCH_DIR/pysite

# --- Model & Test Config ---
MODEL=Qwen/Qwen2.5-Omni-7B
META_PATH=/vllm-omni/seed-tts/dataset/seedtts_testset/zh/meta.lst
STRESS_SCRIPT=/vllm-omni/generalmodeloptimizingqwen3.py

VLLM_PORT=8000
NUM_REQUESTS=200
WARMUP=10
CONCURRENCY=16

CLIENT_TIMEOUT=600

STAGE_INIT_TIMEOUT=1200

OUTPUTS_DIR=$VLLM_WORKDIR/outputs
OUTPUTS_DIR_CTR=/vllm-omni/outputs
COMPARE_JSON=$OUTPUTS_DIR_CTR/sweep_results.json

mkdir -p "$OUTPUTS_DIR"

export HF_TOKEN="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"
export OPENAI_API_KEY=EMPTY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Silence Python 3.12 SyntaxWarnings from third-party libs (e.g. pydub) ────
export PYTHONWARNINGS="ignore::SyntaxWarning"

# ── Install Python deps into /pysite ─────────────────────────────────────────
echo "=== Clearing stale /pysite ==="
rm -rf "$PYSITE"
mkdir -p "$PYSITE"

# ── openai (required) ─────────────────────────────────────────────────────────
echo "=== Installing openai into /pysite ==="
singularity exec --nv \
  -B "$PYSITE":/pysite \
  --env "PYTHONNOUSERSITE=1" \
  "$IMG" \
  pip install openai --target /pysite --quiet

# ── pydantic pin (fixes gradio 5.x conflict) ──────────────────────────────────
echo "=== Pinning pydantic<=2.12.3 into /pysite ==="
singularity exec --nv \
  -B "$PYSITE":/pysite \
  --env "PYTHONNOUSERSITE=1" \
  "$IMG" \
  pip install "pydantic==2.12.3" --target /pysite --quiet

# ── flash-attn (best-effort; non-fatal if the CUDA toolkit version mismatches) ─
echo "=== Installing flash-attn into /pysite (best-effort) ==="
singularity exec --nv \
  -B "$PYSITE":/pysite \
  --env "PYTHONNOUSERSITE=1" \
  "$IMG" \
  bash -c '
    pip install flash-attn \
        --target /pysite \
        --quiet \
        2>/dev/null \
    && echo "  flash-attn: pre-built wheel installed." \
    || {
        echo "  Pre-built wheel unavailable — attempting source build (may take ~10 min)..."
        pip install flash-attn \
            --no-build-isolation \
            --target /pysite \
            --quiet \
            2>/dev/null \
        && echo "  flash-attn: source build installed." \
        || echo "  flash-attn install failed — SDPA fallback will be used (non-fatal)."
    }
  '

# ── Server wrapper (Pure environment) ─────────────────────────────────────────
sing_server() {
    singularity exec --nv \
      -B "$HF_CACHE":/home/users/ntu/es0001an/.cache/huggingface \
      -B "$VLLM_WORKDIR":/vllm-omni \
      -B "$DATASET_DIR":/dataset_generated \
      --env "PYTHONNOUSERSITE=1" \
      --env "PYTHONWARNINGS=$PYTHONWARNINGS" \
      --env "VLLM_USE_V1=1" \
      --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
      --env "HF_TOKEN=$HF_TOKEN" \
      --env "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" \
      "$IMG" "$@"
}

# ── Client wrapper (Injected /pysite) ─────────────────────────────────────────
sing_client() {
    singularity exec --cleanenv --nv \
      -B "$HF_CACHE":/home/users/ntu/es0001an/.cache/huggingface \
      -B "$VLLM_WORKDIR":/vllm-omni \
      -B "$DATASET_DIR":/dataset_generated \
      -B "$PYSITE":/pysite \
      --env "PYTHONPATH=/pysite" \
      --env "PYTHONNOUSERSITE=1" \
      --env "PYTHONWARNINGS=$PYTHONWARNINGS" \
      --env "VLLM_USE_V1=1" \
      --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}" \
      --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
      --env "HF_TOKEN=$HF_TOKEN" \
      "$IMG" "$@"
}

# ── Server management ─────────────────────────────────────────────────────────
SERVER_PID=""

launch_server() {
    local tp=$1 mem=$2
    echo "=== Launching server tp=${tp} mem=${mem} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="

    sing_server vllm serve --omni \
      --model "$MODEL" \
      --tensor-parallel-size "$tp" \
      --gpu-memory-utilization "$mem" \
      --port "$VLLM_PORT" \
      --trust-remote-code \
      --stage-init-timeout "$STAGE_INIT_TIMEOUT" &
    SERVER_PID=$!
    echo "  server PID = $SERVER_PID"
}

wait_for_server() {
    local port=$1 timeout=${2:-1800}
    echo "Waiting for server on port $port (up to ${timeout}s)..."
    sleep 30
    for i in $(seq 1 $timeout); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "Server ready (after ~$((i+30))s)."
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process $SERVER_PID exited unexpectedly." >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: server on port $port did not become ready in time." >&2
    return 1
}

# ── Helper: space-separated physical GPU indices from CUDA_VISIBLE_DEVICES ────
_gpu_ids_from_env() {
    echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' '
}

# ── Helper: fuser + lsof hard-kill for a single GPU index ────────────────────
_hard_kill_gpu() {
    local gpu_idx=$1
    local dev="/dev/nvidia${gpu_idx}"

    if [[ -e "$dev" ]]; then
        echo "  fuser SIGKILL on $dev ..."
        fuser -k "$dev" 2>/dev/null || true
    fi

    local zombie_pids
    zombie_pids=$(lsof -t /dev/shm 2>/dev/null; \
                  lsof -t /dev/nvidia* 2>/dev/null) || true
    zombie_pids=$(echo "$zombie_pids" | sort -u | tr '\n' ' ')

    if [[ -n "${zombie_pids// /}" ]]; then
        echo "  lsof found lingering GPU PIDs: $zombie_pids — sending SIGKILL"
        kill -9 $zombie_pids 2>/dev/null || true
    fi

    sleep 2
}

# ── Robust GPU memory cleanup ─────────────────────────────────────────────────
kill_server() {
    echo "=== Killing server and all GPU workers ==="

    if [[ -n "$SERVER_PID" ]]; then
        echo "  Sending SIGTERM to server PID=$SERVER_PID process group"
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        sleep 5
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  Server still alive — sending SIGKILL"
            kill -9 -- -"$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi

    echo "  Killing Ray worker processes..."
    pkill -9 -f "ray::"        2>/dev/null || true
    pkill -9 -f "ray/workers"  2>/dev/null || true
    pkill -9 -f "_ray_worker"  2>/dev/null || true

    echo "  Killing vLLM/torch worker processes..."
    pkill -9 -f "vllm"                 2>/dev/null || true
    pkill -9 -f "from multiprocessing" 2>/dev/null || true

    ray stop --force 2>/dev/null || true
    sleep 3

    echo "  Running fuser + lsof cleanup on target GPU(s)..."
    for gpu_idx in $(_gpu_ids_from_env); do
        _hard_kill_gpu "$gpu_idx"
    done
    sleep 3

    echo "  Waiting for GPU VRAM to be released..."
    local gpu_list
    gpu_list=$(_gpu_ids_from_env)
    local deadline=$((SECONDS + 180))

    while true; do
        local all_clear=1
        for gpu_idx in $gpu_list; do
            local used_mib
            used_mib=$(nvidia-smi --query-gpu=memory.used \
                         --format=csv,noheader,nounits \
                         --id="$gpu_idx" 2>/dev/null | tr -d ' ')
            if [[ -z "$used_mib" || "$used_mib" -gt 500 ]]; then
                all_clear=0
                echo "  GPU $gpu_idx still has ${used_mib:-?} MiB used — retrying hard kill..."
                _hard_kill_gpu "$gpu_idx"
                break
            fi
        done

        if [[ $all_clear -eq 1 ]]; then
            echo "  GPU(s) $gpu_list are free."
            break
        fi

        if [[ $SECONDS -ge $deadline ]]; then
            echo "  WARNING: GPU(s) $gpu_list did not fully free within 180 s." \
                 "Attempting last-resort user-wide Python kill..." >&2
            pkill -9 -u "$(id -u)" -f python 2>/dev/null || true
            sleep 15
            nvidia-smi --query-gpu=index,memory.used,memory.free \
                       --format=csv,noheader 2>/dev/null || true
            break
        fi
        sleep 5
    done

    sleep 5
    echo "=== kill_server complete ==="
}

run_stress() {
    local label=$1 tp=$2 nm=$3 mem=$4 report=$5
    echo "=== Stress test: $label (client timeout=${CLIENT_TIMEOUT}s) ==="
    sing_client python3 "$STRESS_SCRIPT" \
      --meta "$META_PATH" \
      --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
      --model "$MODEL" \
      --concurrency "$CONCURRENCY" \
      --num-requests "$NUM_REQUESTS" \
      --warmup "$WARMUP" \
      --timeout "$CLIENT_TIMEOUT" \
      --config-label "$label" \
      --tp "$tp" \
      --num-models-on-card "$nm" \
      --gpu-mem-util "$mem" \
      --report-json "$report" \
      --compare-json "$COMPARE_JSON" \
      --disable-omni-sampling
}


# ── tp=1 on GPU 0 ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

launch_server 1 0.90
if wait_for_server $VLLM_PORT; then run_stress "tp1_gpu0_mem90" 1 1 0.90 "$OUTPUTS_DIR_CTR/tp1_gpu0_mem90.json"; fi
kill_server

launch_server 1 0.80
if wait_for_server $VLLM_PORT; then run_stress "tp1_gpu0_mem80" 1 1 0.80 "$OUTPUTS_DIR_CTR/tp1_gpu0_mem80.json"; fi
kill_server

launch_server 1 0.70
if wait_for_server $VLLM_PORT; then run_stress "tp1_gpu0_mem70" 1 1 0.70 "$OUTPUTS_DIR_CTR/tp1_gpu0_mem70.json"; fi
kill_server

launch_server 1 0.60
if wait_for_server $VLLM_PORT; then run_stress "tp1_gpu0_mem60" 1 1 0.60 "$OUTPUTS_DIR_CTR/tp1_gpu0_mem60.json"; fi
kill_server

# ── tp=1 on GPU 1 ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

launch_server 1 0.90
if wait_for_server $VLLM_PORT; then run_stress "tp1_gpu1_mem90" 1 1 0.90 "$OUTPUTS_DIR_CTR/tp1_gpu1_mem90.json"; fi
kill_server

# ── tp=2 across GPUs 0+1 ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1

launch_server 2 0.90
if wait_for_server $VLLM_PORT; then run_stress "tp2_gpu01_mem90" 2 1 0.90 "$OUTPUTS_DIR_CTR/tp2_gpu01_mem90.json"; fi
kill_server

launch_server 2 0.80
if wait_for_server $VLLM_PORT; then run_stress "tp2_gpu01_mem80" 2 1 0.80 "$OUTPUTS_DIR_CTR/tp2_gpu01_mem80.json"; fi
kill_server

# ── tp=2 across GPUs 1+2 ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1,2

launch_server 2 0.90
if wait_for_server $VLLM_PORT; then run_stress "tp2_gpu12_mem90" 2 1 0.90 "$OUTPUTS_DIR_CTR/tp2_gpu12_mem90.json"; fi
kill_server

echo "=== Sweep complete. Results: $OUTPUTS_DIR/sweep_results.json ==="