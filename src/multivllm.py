import os
import traceback
import time
import json
import csv
import glob
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import torch

from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

# Configuration
SEED = 42
DATASET_DIR = "/dataset_generated"
OUTPUT_DIR = "./outputs"

class MetricsTracker:
    """Track TTFT, TBT, and stage-specific timing metrics"""
    
    def __init__(self):
        self.metrics = []
        self.stage_start_times = {}  # Track start time per stage
    
    def analyze_output(self, output, stage_idx: int, audio_file: str, stage_name: str) -> Dict[str, Any]:
        """Extract timing and quality metrics from vLLM output"""
        current_time = time.time()
        
        metrics = {
            "audio_file": os.path.basename(audio_file),
            "request_id": output.request_id,
            "stage": stage_name,
            "stage_index": stage_idx,
            "timestamp": current_time,
        }
        
        # Calculate stage-specific latency
        if stage_idx in self.stage_start_times:
            metrics["stage_latency"] = current_time - self.stage_start_times[stage_idx]
        
        # Extract token-level timing if available
        if hasattr(output, 'metrics') and output.metrics:
            metrics.update({
                "ttft": output.metrics.get('time_to_first_token_s', None),
                "tbt_mean": output.metrics.get('mean_time_between_tokens_s', None),
                "tbt_p50": output.metrics.get('p50_time_between_tokens_s', None),
                "tbt_p90": output.metrics.get('p90_time_between_tokens_s', None),
            })
        
        # Token counts
        if hasattr(output, 'outputs') and len(output.outputs) > 0:
            output_obj = output.outputs[0]
            metrics["output_tokens"] = len(output_obj.token_ids) if hasattr(output_obj, 'token_ids') else 0
            metrics["output_text_length"] = len(output_obj.text) if hasattr(output_obj, 'text') else 0
        else:
            metrics["output_tokens"] = 0
            metrics["output_text_length"] = 0
        
        return metrics
    
    def save_metrics_csv(self, filename: str):
        """Save all metrics to CSV"""
        if not self.metrics:
            return
        
        headers = [
            "audio_file", "request_id", "stage", "stage_index", "stage_latency",
            "ttft", "tbt_mean", "tbt_p50", "tbt_p90", 
            "output_tokens", "output_text_length", "status"
        ]
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                for metric in self.metrics:
                    writer.writerow(metric)
            print(f"\n[METRICS] Saved to: {filename}")
        except Exception as e:
            print(f"[ERROR] Could not save metrics: {e}")
    
    def save_summary_csv(self, filename: str):
        """Save per-file summary with thinker/talker/code2wav times"""
        if not self.metrics:
            return
        
        # Group by audio file
        file_metrics = defaultdict(lambda: {
            "thinker_latency": None,
            "talker_latency": None,
            "code2wav_latency": None,
            "thinker_tokens": 0,
            "talker_tokens": 0,
            "total_latency": 0
        })
        
        for metric in self.metrics:
            file_key = metric["audio_file"]
            stage = metric["stage"]
            
            if "thinker" in stage.lower():
                file_metrics[file_key]["thinker_latency"] = metric.get("stage_latency")
                file_metrics[file_key]["thinker_tokens"] = metric.get("output_tokens", 0)
            elif "talker" in stage.lower():
                file_metrics[file_key]["talker_latency"] = metric.get("stage_latency")
                file_metrics[file_key]["talker_tokens"] = metric.get("output_tokens", 0)
            elif "code2wav" in stage.lower():
                file_metrics[file_key]["code2wav_latency"] = metric.get("stage_latency")
        
        # Calculate total latency
        for file_key in file_metrics:
            fm = file_metrics[file_key]
            total = 0
            for lat_key in ["thinker_latency", "talker_latency", "code2wav_latency"]:
                if fm[lat_key] is not None:
                    total += fm[lat_key]
            fm["total_latency"] = total
        
        # Write summary CSV
        headers = [
            "audio_file", "thinker_latency", "thinker_tokens",
            "talker_latency", "talker_tokens", 
            "code2wav_latency", "total_latency"
        ]
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for file_key in sorted(file_metrics.keys()):
                    row = {"audio_file": file_key}
                    row.update(file_metrics[file_key])
                    writer.writerow(row)
            print(f"[SUMMARY] Saved to: {filename}")
        except Exception as e:
            print(f"[ERROR] Could not save summary: {e}")

def find_audio_files(directory: str) -> List[str]:
    print(f"\n[SCAN] Looking for audio files in: {directory}")
    if not os.path.exists(directory):
        print(f"  [ERROR] Directory not found: {directory}")
        return []
    
    audio_files = []
    for ext in ['*.mp3', '*.flac', '*.wav']:
        found = glob.glob(os.path.join(directory, ext))
        audio_files.extend(found)
    
    audio_files.sort()
    print(f"  [OK] Found {len(audio_files)} audio files")
    return audio_files

def prepare_prompt(audio_path: str):
    """Load audio and create a single prompt dictionary."""
    try:
        audio_data, sample_rate = sf.read(audio_path)
        
        # Make mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        audio_data = audio_data.astype(np.float32)
        
        # Validation
        if len(audio_data) < sample_rate * 0.1:
            print(f"  [SKIP] {os.path.basename(audio_path)} (Too short)")
            return None
            
        prompt = {
            "prompt": (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                "Describe this audio in detail.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "multi_modal_data": {"audio": (audio_data, sample_rate)},
        }
        return prompt
    except Exception as e:
        print(f"  [ERROR] Loading {audio_path}: {e}")
        return None

def main() -> None:
    print("=== vLLM-Omni Batch Processing (Thinker/Talker Separation) ===")
    
    # Setup
    tracker = MetricsTracker()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find files
    audio_files = find_audio_files(DATASET_DIR)
    if not audio_files:
        return

    # 1. Prepare Prompts
    print("\n=== Pre-loading Audio Data ===")
    valid_prompts = []
    valid_audio_paths = []
    
    for path in audio_files:
        p = prepare_prompt(path)
        if p:
            valid_prompts.append(p)
            valid_audio_paths.append(path)
            
    print(f"Ready to process {len(valid_prompts)} files.")

    # 2. Initialize Engine
    print("\n=== Initializing OmniLLM ===")
    device_mapping = [[0], [1], [1]]
    
    omni_llm = OmniLLM(
        model="Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        dtype="bfloat16",
        runtime={"devices": device_mapping},
        init_sleep_seconds=10,
        max_model_len=2048,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )

    # Sampling Params
    # Stage 0: Thinker, Stage 1: Talker, Stage 2: Code2Wav
    sampling_params_list = [
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.1),  # Thinker
        SamplingParams(temperature=0.9, top_p=0.8, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.05, stop_token_ids=[8294]),  # Talker
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.1),  # Code2Wav
    ]
    
    stage_names = ["thinker", "talker", "code2wav"]

    # 3. Generate
    print("\n=== Generating (Batch Mode) ===")
    batch_start = time.time()
    
    try:
        results_list = omni_llm.generate(valid_prompts, sampling_params_list)
        
        print("\n=== Saving Results ===")
        
        # Track output counts per file
        file_stage_counters = defaultdict(lambda: defaultdict(int))
        
        for stage_idx, stage_outputs in enumerate(results_list):
            output_type = stage_outputs.final_output_type
            stage_name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage_{stage_idx}"
            
            # Record stage start time
            tracker.stage_start_times[stage_idx] = time.time()
            
            print(f"\n[STAGE {stage_idx}] Processing {stage_name} ({output_type})...")
            
            # Iterate through all requests in this stage-output batch
            for output in stage_outputs.request_output:
                req_idx = int(output.request_id)
                
                if req_idx >= len(valid_audio_paths):
                    continue
                
                audio_path = valid_audio_paths[req_idx]
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                
                # Increment counter for this specific file and stage
                file_stage_counters[req_idx][stage_name] += 1
                stage_count = file_stage_counters[req_idx][stage_name]
                
                # Determine label (text or audio)
                output_label = "text" if output_type == "text" else "audio"
                full_stage_name = f"{stage_name}_{output_label}"
                
                if output_type == "text":
                    text_content = output.outputs[0].text
                    
                    # Metrics
                    metrics = tracker.analyze_output(output, stage_idx, audio_path, full_stage_name)
                    metrics["status"] = "success"
                    tracker.metrics.append(metrics)
                    
                    # Save File
                    filename = f"{audio_basename}_{stage_name}_{stage_count}.txt"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    
                elif output_type == "audio":
                    audio_tensor = output.multimodal_output["audio"]
                    
                    # Metrics
                    metrics = tracker.analyze_output(output, stage_idx, audio_path, full_stage_name)
                    metrics["status"] = "success"
                    tracker.metrics.append(metrics)
                    
                    # Save File
                    filename = f"{audio_basename}_{stage_name}_{stage_count}.wav"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    sf.write(filepath, audio_tensor.detach().cpu().numpy(), samplerate=24000)
            
            print(f"[STAGE {stage_idx}] Completed {stage_name}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
        
    finally:
        total_time = time.time() - batch_start
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete in {total_time:.2f}s")
        print(f"{'='*60}")
        
        # Save detailed metrics
        tracker.save_metrics_csv(os.path.join(OUTPUT_DIR, "batch_metrics_detailed.csv"))
        
        # Save summary with thinker/talker/code2wav breakdown
        tracker.save_summary_csv(os.path.join(OUTPUT_DIR, "batch_metrics_summary.csv"))
        
        omni_llm.close()

if __name__ == "__main__":
    main()