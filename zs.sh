#!/bin/bash
#PBS -N vllm_omni_infer
#PBS -l select=1:ncpus=24:ngpus=3:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -P personal-es0001an

module load singularity
set -euo pipefail
cd "$PBS_O_WORKDIR"

SCRATCH_DIR=/scratch/users/ntu/es0001an
IMG=$SCRATCH_DIR/vllm-omni-try/vllm-omni-official.sif
HF_CACHE=$SCRATCH_DIR/hf_cache
VLLM_WORKDIR=$SCRATCH_DIR/vllm-omni-try
DATASET_DIR=$SCRATCH_DIR/dataset_generated
PYSITE=$SCRATCH_DIR/pysite

# --- Model & Test Config ---
MODEL=Qwen/Qwen2.5-Omni-7B
META_PATH=/vllm-omni/seed-tts/dataset/seedtts_testset/zh/meta.lst
INFER_SCRIPT=/vllm-omni/zeroshottts.py

VLLM_PORT=8000
OUTPUT_DIR=/vllm-omni/outputs/infer_wavs
PROFILE_JSON=/vllm-omni/outputs/profile.json
PROFILE_CSV=/vllm-omni/outputs/profile.csv

export HF_TOKEN="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"
export OPENAI_API_KEY=EMPTY

mkdir -p "$SCRATCH_DIR/vllm-omni-try/outputs"

echo "=== Clearing stale /pysite ==="
rm -rf "$PYSITE"
mkdir -p "$PYSITE"

echo "=== Installing openai into /pysite ==="
singularity exec --nv \
  -B "$PYSITE":/pysite \
  --env "PYTHONNOUSERSITE=1" \
  "$IMG" \
  pip install openai --target /pysite --quiet || true

# ── Server wrapper (Pure environment) ─────────────────────────────────────────
sing_server() {
    singularity exec --cleanenv --nv \
      -B "$HF_CACHE":/home/users/ntu/es0001an/.cache/huggingface \
      -B "$VLLM_WORKDIR":/vllm-omni \
      -B "$DATASET_DIR":/dataset_generated \
      --env "PYTHONNOUSERSITE=1" \
      --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}" \
      --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
      --env "HF_TOKEN=$HF_TOKEN" \
      --env "CUDA_LAUNCH_BLOCKING=1" \
      --env "VLLM_ATTENTION_BACKEND=FLASH_ATTN" \
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

    sing_server python3 -m vllm_omni.entrypoints.cli.main serve \
      --omni \
      --model "$MODEL" \
      --tensor-parallel-size "$tp" \
      --gpu-memory-utilization "$mem" \
      --port "$VLLM_PORT" \
      --trust-remote-code \
      --served-model-name "$MODEL" \
      --max-model-len 4096 &
      
    SERVER_PID=$!
    echo "  server PID = $SERVER_PID"
}

wait_for_server() {
    local port=$1 timeout=${2:-600}
    echo "Waiting for server on port $port (up to ${timeout}s)..."
    sleep 15
    for i in $(seq 1 $timeout); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "Server ready (after ~$((i+15))s)."
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process $SERVER_PID exited unexpectedly." >&2
            return 1
        fi
        sleep 2
    done
    echo "ERROR: server on port $port did not become ready in time." >&2
    return 1
}

kill_server() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "=== Killing server PID=$SERVER_PID ==="
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    pkill -f "ray::" 2>/dev/null || true
    sleep 5
}

# ── Run inference ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2
launch_server 1 0.90

if wait_for_server $VLLM_PORT; then
    echo "=== Running inference ==="
    sing_client python3 "$INFER_SCRIPT" \
      --meta "$META_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
      --model "$MODEL" \
      --modalities "audio" \
      --temperature 0.2 \
      --num-retries 3 \
      --retry-sleep 2.0 \
      --overwrite \
      --profile-stage-breakdown \
      --profile-output-json "$PROFILE_JSON" \
      --profile-output-csv "$PROFILE_CSV"
else
    echo "=== Skipping inference due to server failure ==="
fi

kill_server

echo "=== Inference complete. Wavs in $OUTPUT_DIR ==="
echo "=== Profile JSON: $PROFILE_JSON ==="
echo "=== Profile CSV:  $PROFILE_CSV ==="