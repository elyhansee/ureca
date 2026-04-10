#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -N omni_pipeline_trace
#PBS -P personal-es0001an

set -euo pipefail
source activate  /home/users/ntu/es0001an/scratch/vllm_speech_translation/vllm_env

# --- CONFIG ---
scratch="/scratch/users/ntu/$USER"
project_folder="$scratch/gpt-oss"
whisper_model="$scratch/whisper/model_weights"
gpt_model="$project_folder/model_weights"
sif_image="$project_folder/vllm.sif"
libs_folder="$scratch/omni_libs"

fake_home="$project_folder/omni_home"
mkdir -p "$fake_home"

cd "$PBS_O_WORKDIR"

module load singularity
module load cuda

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_VLLM_TORCH_COMPILE_DISABLE=1
export SINGULARITYENV_TORCH_COMPILE_DISABLE=1
export SINGULARITYENV_PYTHONPATH="/opt/omni_libs${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================================"
echo "STARTING OMNI-PIPELINE (Sequential Loading - Memory Fix)"
echo "========================================================"

wait_for_server() {
    port=$1
    name=$2
    log_file=$3
    max_wait=1200
    echo "Waiting for $name on port $port..."
    for i in $(seq 1 $max_wait); do
        if [ -n "${4:-}" ] && ! ps -p $4 > /dev/null 2>&1; then
            echo "✗ $name process died unexpectedly!"
            echo "Last 50 lines of log:"
            tail -n 50 "$log_file"
            return 1
        fi
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "✓ $name is Ready after $i seconds!"
            return 0
        fi
        if [ $((i % 30)) -eq 0 ]; then 
            echo "  ...still loading $name ($i s elapsed)"
        fi
        sleep 1
    done
    echo "✗ $name timed out after $max_wait seconds."
    tail -n 50 "$log_file"
    return 1
}

echo ">> Launching Whisper FIRST (Port 8000) with reduced memory..."
singularity exec --nv \
  -B "$fake_home:$HOME" \
  -B "$whisper_model:/model" \
  -B "$libs_folder:/opt/omni_libs" \
  "$sif_image" \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name whisper-large-v3 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 448 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager \
  --dtype float16 \
  > whisper.log 2>&1 &
PID_WHISPER=$!

wait_for_server 8000 "Whisper" "whisper.log" $PID_WHISPER || exit 1

echo ""
echo ">> Now launching GPT-OSS (Port 8001) with remaining memory..."
singularity exec --nv \
  -B "$fake_home:$HOME" \
  -B "$gpt_model:/model" \
  -B "$libs_folder:/opt/omni_libs" \
  "$sif_image" \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name gpt-oss \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.65 \
  --enforce-eager \
  --trust-remote-code \
  --tokenizer-mode auto \
  > gpt.log 2>&1 &
PID_GPT=$!

wait_for_server 8001 "GPT-OSS" "gpt.log" $PID_GPT || { kill $PID_WHISPER 2>/dev/null || true; exit 1; }

echo ""
echo "========================================================"
echo "WHISPER DEBUG TEST"
echo "========================================================"

singularity exec --nv --network host \
  -B "$fake_home:$HOME" \
  -B "$PWD:/code" \
  -B "$libs_folder:/opt/omni_libs" \
  --pwd /code \
  "$sif_image" \
  python3 test_whisper.py

echo ""
echo "========================================================"
echo "RUNNING TRACES"
echo "========================================================"

singularity exec --nv --network host \
  -B "$fake_home:$HOME" \
  -B "$PWD:/code" \
  -B "$libs_folder:/opt/omni_libs" \
  --pwd /code \
  "$sif_image" \
  python3 pipeline_omni.py

RET=$?

echo "Stopping servers..."
kill $PID_WHISPER 2>/dev/null || true
kill $PID_GPT 2>/dev/null || true
wait 2>/dev/null || true

exit $RET
