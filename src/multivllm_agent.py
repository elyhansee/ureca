#!/usr/bin/env python3
"""
vLLM-Omni Pipeline with Function Calling - TRUE BATCH VERSION
Two-pass: batch planning → tool execution → batch synthesis
"""

import os
import json
import time
import csv
import glob
import traceback
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import soundfile as sf
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

SEED = 42
DATASET_DIR = "/dataset_generated"
OUTPUT_DIR = "./outputs"
TRACE_FILE = "./omni_vllm_traces.csv"

SYSTEM_PROMPT = """You are a helpful assistant. Use lookup_customer if a phone number is provided.

When you need to call a function, respond ONLY with a JSON object:
{"function_call": {"name": "lookup_customer", "arguments": {"phone_number": "+1234567890"}}}

After receiving tool results, provide a natural response."""


SAMPLING_PARAMS = [
    SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED,
                   detokenize=True, repetition_penalty=1.1),           # Thinker
    SamplingParams(temperature=0.9, top_p=0.8, max_tokens=2048, seed=SEED,
                   detokenize=True, repetition_penalty=1.05,
                   stop_token_ids=[8294]),                              # Talker
    SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED,
                   detokenize=True, repetition_penalty=1.1),            # Code2Wav
]


def execute_tool_call(tool_name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid arguments"})
    if tool_name == "lookup_customer":
        return json.dumps({
            "status": "success",
            "name": "Esther Lee",
            "plan": "Premium",
            "balance": "$0.00",
            "phone": arguments.get("phone_number", "Unknown")
        })
    return json.dumps({"error": "Tool not found"})

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    text = text.strip()
    if "function_call" in text:
        try:
            parsed = json.loads(text)
            fc = parsed["function_call"]
            args = fc["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            return fc["name"], args
        except (json.JSONDecodeError, KeyError):
            pass
    if text.startswith('{') and "phone_number" in text:
        try:
            parsed = json.loads(text)
            if "phone_number" in parsed:
                return "lookup_customer", parsed
        except json.JSONDecodeError:
            pass
    phone_pattern = r'(?:\+\d{1,3}[\s-]?)?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}'
    match = re.search(phone_pattern, text)
    if match and any(k in text.lower() for k in ['lookup', 'search', 'find', 'customer']):
        return "lookup_customer", {"phone_number": match.group().strip()}
    return None


class TraceRecorder:
    HEADERS = [
        "TraceID", "Timestamp", "Status", "TotalLatency(s)",
        "AudioInput", "Transcription", "ASR_Latency(s)",
        "LLM_Plan_Latency(s)", "Tool_Name", "Tool_Args",
        "Tool_Exec_Latency(s)", "LLM_Synth_Latency(s)",
        "Final_Response_Len", "Output_Tokens", "Stage"
    ]

    def __init__(self, filename: str):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerow(self.HEADERS)

    def log(self, data: Dict):
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([data.get(h, "") for h in self.HEADERS])

def load_audio(path: str) -> Optional[Tuple[np.ndarray, int]]:
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if len(data) < sr * 0.1:
            return None
        return data, sr
    except Exception as e:
        print(f"  [ERROR] Loading {path}: {e}")
        return None

def make_initial_prompt(audio_data: np.ndarray, sr: int) -> Dict:
    return {
        "prompt": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "What can I help you with?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"audio": (audio_data, sr)}
    }

def make_synth_prompt(audio_data: np.ndarray, sr: int,
                      plan_text: str, tool_result: str) -> Dict:
    return {
        "prompt": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "What can I help you with?<|im_end|>\n"
            f"<|im_start|>assistant\n{plan_text}<|im_end|>\n"
            f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"audio": (audio_data, sr)}
    }


def run_batch_pipeline():
    print("=== vLLM-Omni Function Calling Pipeline (Batched) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recorder = TraceRecorder(TRACE_FILE)

    print(f"\n[SCAN] Looking for audio in: {DATASET_DIR}")
    audio_files = sorted(
        f for ext in ['*.mp3', '*.flac', '*.wav']
        for f in glob.glob(os.path.join(DATASET_DIR, ext))
    )
    print(f"  Found {len(audio_files)} files")
    if not audio_files:
        return

    print("\n[LOAD] Pre-loading audio...")
    items = []   # list of (file_id, path, audio_data, sr)
    for idx, path in enumerate(audio_files, 1):
        result = load_audio(path)
        if result:
            items.append((f"{idx:03d}", path, result[0], result[1]))
        else:
            print(f"  [SKIP] {os.path.basename(path)}")

    print(f"  Loaded {len(items)} valid files")

    print("\n[INIT] Loading vLLM-Omni...")
    omni_llm = OmniLLM(
        model="Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        dtype="bfloat16",
        runtime={"devices": [[0], [1], [1]]},
        init_sleep_seconds=10,
        max_model_len=2048,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )

    t_start = time.time()

    try:
        print(f"\n[PASS 1] Planning batch ({len(items)} prompts)...")
        plan_prompts = [make_initial_prompt(d, sr) for _, _, d, sr in items]

        t_plan = time.time()
        plan_results = omni_llm.generate(plan_prompts, SAMPLING_PARAMS)
        plan_latency = time.time() - t_plan
        print(f"  Planning done in {plan_latency:.1f}s")

        plan_texts = {}   # idx -> text
        for stage_out in plan_results:
            if stage_out.final_output_type == "text":
                for out in stage_out.request_output:
                    req_idx = int(out.request_id)
                    plan_texts[req_idx] = out.outputs[0].text

        tool_needed = []   
        for req_idx, (file_id, path, audio, sr) in enumerate(items):
            plan_text = plan_texts.get(req_idx, "")
            basename = os.path.splitext(os.path.basename(path))[0]

            tool_call = parse_tool_call(plan_text)

            if tool_call:
                tool_name, tool_args = tool_call
                print(f"  [{file_id}] Tool call: {tool_name}({tool_args})")
                t_tool = time.time()
                tool_result = execute_tool_call(tool_name, tool_args)
                tool_latency = time.time() - t_tool
                tool_needed.append((req_idx, file_id, path, audio, sr,
                                    plan_text, tool_name, tool_args,
                                    tool_result, tool_latency))
            else:
                out_path = os.path.join(OUTPUT_DIR, f"{basename}_direct.txt")
                with open(out_path, "w") as f:
                    f.write(plan_text)
                recorder.log({
                    "TraceID": file_id,
                    "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "AudioInput": os.path.basename(path),
                    "LLM_Plan_Latency(s)": round(plan_latency / len(items), 3),
                    "Transcription": plan_text[:200],
                    "Final_Response_Len": len(plan_text),
                    "Stage": "direct",
                    "Status": "Success",
                })

        print(f"\n  {len(tool_needed)} files need tool synthesis")
        print(f"  {len(items) - len(tool_needed)} files completed directly")

        if tool_needed:
            print(f"\n[PASS 2] Synthesis batch ({len(tool_needed)} prompts)...")
            synth_prompts = [
                make_synth_prompt(audio, sr, plan_text, tool_result)
                for _, _, _, audio, sr, plan_text, _, _, tool_result, _ in tool_needed
            ]

            t_synth = time.time()
            synth_results = omni_llm.generate(synth_prompts, SAMPLING_PARAMS)
            synth_latency = time.time() - t_synth
            print(f"  Synthesis done in {synth_latency:.1f}s")

            synth_texts = {}
            synth_audios = {}
            for stage_out in synth_results:
                for out in stage_out.request_output:
                    req_idx = int(out.request_id)
                    if stage_out.final_output_type == "text":
                        synth_texts[req_idx] = out.outputs[0].text
                    elif stage_out.final_output_type == "audio":
                        synth_audios[req_idx] = out.multimodal_output["audio"]

            # Save outputs
            for synth_idx, (_, file_id, path, audio, sr,
                             plan_text, tool_name, tool_args,
                             tool_result, tool_latency) in enumerate(tool_needed):
                basename = os.path.splitext(os.path.basename(path))[0]

                final_text = synth_texts.get(synth_idx, "")
                if final_text:
                    txt_path = os.path.join(OUTPUT_DIR, f"{basename}_synthesis.txt")
                    with open(txt_path, "w") as f:
                        f.write(final_text)

                if synth_idx in synth_audios:
                    wav_path = os.path.join(OUTPUT_DIR, f"{basename}_response.wav")
                    sf.write(wav_path,
                             synth_audios[synth_idx].detach().cpu().numpy(),
                             samplerate=24000)

                recorder.log({
                    "TraceID": file_id,
                    "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "AudioInput": os.path.basename(path),
                    "LLM_Plan_Latency(s)": round(plan_latency / len(items), 3),
                    "Tool_Name": tool_name,
                    "Tool_Args": json.dumps(tool_args),
                    "Tool_Exec_Latency(s)": round(tool_latency, 3),
                    "LLM_Synth_Latency(s)": round(synth_latency / len(tool_needed), 3),
                    "Transcription": plan_text[:200],
                    "Final_Response_Len": len(final_text),
                    "Stage": "tool_synthesis",
                    "Status": "Success",
                })

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()

    finally:
        total = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"Done in {total:.1f}s ({total/60:.1f} min)")
        print(f"Traces: {TRACE_FILE}")
        print(f"Outputs: {OUTPUT_DIR}")
        print(f"{'='*60}")
        omni_llm.close()


if __name__ == "__main__":
    run_batch_pipeline()