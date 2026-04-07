#!/bin/bash
#PBS -N vllm_omni_test
#PBS -l select=1:ncpus=24:ngpus=3:mem=64gb
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -P personal-es0001an

set -euo pipefail
cd "$PBS_O_WORKDIR"

export CUDA_VISIBLE_DEVICES=0,1,2
export VLLM_USE_V1=1

module load singularity

SCRATCH_DIR=/scratch/users/ntu/es0001an
IMG=$SCRATCH_DIR/vllm-omni-try/vllm-omni.sif

echo "=== Running vLLM-Omni Test ==="
echo "Node: $(hostname)"

singularity exec --nv \
  -B $SCRATCH_DIR/hf_cache:/home/users/ntu/es0001an/.cache/huggingface \
  -B $SCRATCH_DIR/vllm-omni-try:/vllm-omni \
  --env PYTHONNOUSERSITE=1 \
  --env VLLM_USE_V1=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2 \
  "$IMG" \
  python3 /vllm-omni/test_mini.py

echo "=== Job Complete ==="