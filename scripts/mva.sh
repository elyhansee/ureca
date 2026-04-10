#!/bin/bash
#PBS -N vllm_omni_agent_batch-FAISS
#PBS -l select=1:ncpus=24:ngpus=3:mem=64gb
#PBS -l walltime=04:00:00 
#PBS -j oe
#PBS -P personal-es0001an

set -euo pipefail
cd "$PBS_O_WORKDIR"

export CUDA_VISIBLE_DEVICES=0,1,2
export VLLM_USE_V1=1

module load singularity

SCRATCH_DIR=/scratch/users/ntu/es0001an
IMG=$SCRATCH_DIR/vllm-omni-try/vllm-omni.sif

export PYTHONUSERBASE=$SCRATCH_DIR/.local
export PATH=$PYTHONUSERBASE/bin:$PATH

echo "=== Installing Dependencies (FAISS & Sentence Transformers) ==="
singularity exec --nv \
  -B $SCRATCH_DIR:/scratch/users/ntu/es0001an \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  "$IMG" \
  pip install --user faiss-cpu sentence-transformers

echo "=== Running vLLM Agent ==="
singularity exec --nv \
  -B $SCRATCH_DIR/hf_cache:/home/users/ntu/es0001an/.cache/huggingface \
  -B $SCRATCH_DIR/vllm-omni-try:/vllm-omni \
  -B $SCRATCH_DIR/dataset_generated:/dataset_generated \
  -B $PYTHONUSERBASE:$PYTHONUSERBASE \
  --env PYTHONUSERBASE=$PYTHONUSERBASE \
  --env PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages \
  --env VLLM_USE_V1=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2 \
  "$IMG" \
  python3 /vllm-omni/multivllm_faiss.py

echo "=== Job Complete ==="