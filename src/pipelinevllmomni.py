#!/usr/bin/env python3
"""
vLLM-Omni Pipeline with Function Calling
Similar architecture to pipeline_omni.py but adapted for vLLM-Omni batch processing
"""

import os
import json
import time
import csv
import glob
import traceback
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import soundfile as sf
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

SEED = 42
DATASET_DIR = "/dataset_generated"
OUTPUT_DIR = "./outputs"
TRACE_FILE = "./omni_vllm_traces.csv"

TOOLS_DEFINITION = [{
    "type": "function",
    "function": {
        "name": "lookup_customer",
        "description": "Look up a customer by phone number",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "The customer's phone number"}
            },
            "required": ["phone_number"]
        }
    }
}]

SYSTEM_PROMPT = """You are a helpful assistant. Use lookup_customer if a phone number is provided.

When you need to call a function, respond ONLY with a JSON object:
{"function_call": {"name": "lookup_customer", "arguments": {"phone_number": "+1234567890"}}}

After receiving tool results, provide a natural response."""

def execute_tool_call(tool_name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid arguments format"})
    
    if tool_name == "lookup_customer":
        phone = arguments.get("phone_number", "Unknown")
        result = {
            "status": "success",
            "name": "Esther Lee",
            "plan": "Premium",
            "balance": "$0.00",
            "phone": phone
        }
        return json.dumps(result)
    return json.dumps({"error": "Tool not found"})

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    text = text.strip()
    
    # Method 1: Explicit JSON function call
    if "function_call" in text:
        try:
            parsed = json.loads(text)
            if "function_call" in parsed:
                fc = parsed["function_call"]
                args = fc["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                return fc["name"], args
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Method 2: Direct JSON (lazy model behavior)
    if text.startswith('{') and "phone_number" in text:
        try:
            parsed = json.loads(text)
            if "phone_number" in parsed:
                return "lookup_customer", parsed
        except json.JSONDecodeError:
            pass
    
    # Method 3: Phone number pattern matching (fallback)
    phone_pattern = r'(?:\+\d{1,3}[\s-]?)?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}'
    match = re.search(phone_pattern, text)
    if match and any(keyword in text.lower() for keyword in ['lookup', 'search', 'find', 'customer', 'database']):
        return "lookup_customer", {"phone_number": match.group().strip()}
    
    return None

class TraceRecorder:
    def __init__(self, filename: str):
        self.filename = filename
        self.headers = [
            "TraceID", "Timestamp", "Status", "TotalLatency(s)",
            "AudioInput", "Transcription", "ASR_Latency(s)",
            "LLM_Plan_Latency(s)", "Tool_Name", "Tool_Args",
            "Tool_Exec_Latency(s)", "LLM_Synth_Latency(s)",
            "Final_Response_Len", "TTS_Latency(s)",
            "Output_Tokens", "Stage"
        ]
        
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                csv.writer(f).writerow(self.headers)
    
    def log(self, data: Dict):
        with open(self.filename, mode='a', newline='') as f:
            row = [data.get(h, "") for h in self.headers]
            csv.writer(f).writerow(row)


def load_audio(audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
    try:
        audio_data, sample_rate = sf.read(audio_path)
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        audio_data = audio_data.astype(np.float32)
        
        if len(audio_data) < sample_rate * 0.1:
            return None
            
        return audio_data, sample_rate
    except Exception as e:
        print(f"  [ERROR] Loading {audio_path}: {e}")
        return None

def create_prompt(audio_data: np.ndarray, sample_rate: int, 
                  conversation_history: Optional[str] = None) -> Dict:
    
    if conversation_history:
        prompt_text = conversation_history
    else:
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "What can I help you with?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"audio": (audio_data, sample_rate)}
    }

def process_single_item(
    omni_llm: OmniLLM,
    audio_path: str,
    file_id: str,
    sampling_params_list: List[SamplingParams],
    recorder: TraceRecorder
) -> bool:
    """
    Process one audio file through the complete pipeline
    Returns: True if successful
    """
    
    trace = {
        "TraceID": file_id,
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "AudioInput": os.path.basename(audio_path),
        "Status": "Failed"
    }
    
    t_total_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_id}")
    print(f"{'='*60}")
    
    t_asr_start = time.time()
    audio_result = load_audio(audio_path)
    if not audio_result:
        recorder.log(trace)
        return False
    
    audio_data, sample_rate = audio_result
    trace["ASR_Latency(s)"] = round(time.time() - t_asr_start, 3)
    
    print("  [STAGE 1] Planning...")
    t_plan_start = time.time()
    
    initial_prompt = create_prompt(audio_data, sample_rate)
    
    try:
        results = omni_llm.generate([initial_prompt], sampling_params_list)
        trace["LLM_Plan_Latency(s)"] = round(time.time() - t_plan_start, 3)
        
        first_stage = results[0]
        if first_stage.final_output_type != "text":
            print("  [WARNING] Expected text output in planning stage")
            trace["Status"] = "Unexpected output type"
            recorder.log(trace)
            return False
        
        plan_output = first_stage.request_output[0]
        plan_text = plan_output.outputs[0].text
        trace["Transcription"] = plan_text[:200]  # Store snippet
        
        print(f"  Plan Output: {plan_text[:150]}...")
        print(f"  [DEBUG] Full plan text: {repr(plan_text)}")  # Debug output
        
        tool_call = parse_tool_call(plan_text)
        
        if tool_call:
            tool_name, tool_args = tool_call
            print(f"  [TOOL TRIGGERED] {tool_name}({tool_args})")
            
            trace["Tool_Name"] = tool_name
            trace["Tool_Args"] = json.dumps(tool_args)
            
            t_tool_start = time.time()
            try:
                tool_result = execute_tool_call(tool_name, tool_args)
                trace["Tool_Exec_Latency(s)"] = round(time.time() - t_tool_start, 3)
                print(f"  Tool Result: {tool_result}")
            except Exception as tool_err:
                print(f"  [ERROR] Tool execution failed: {tool_err}")
                tool_result = json.dumps({"error": str(tool_err)})
                trace["Tool_Exec_Latency(s)"] = round(time.time() - t_tool_start, 3)
            
            print("  [STAGE 2] Synthesizing response...")
            t_synth_start = time.time()
            
            conversation = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                "What can I help you with?<|im_end|>\n"
                f"<|im_start|>assistant\n{plan_text}<|im_end|>\n"
                f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            synth_prompt = create_prompt(audio_data, sample_rate, conversation)
            synth_results = omni_llm.generate([synth_prompt], sampling_params_list)
            
            trace["LLM_Synth_Latency(s)"] = round(time.time() - t_synth_start, 3)
            
            for stage_output in synth_results:
                output_type = stage_output.final_output_type
                
                for output in stage_output.request_output:
                    if output_type == "text":
                        final_text = output.outputs[0].text
                        trace["Final_Response_Len"] = len(final_text)
                        trace["Output_Tokens"] = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
                        
                        text_path = os.path.join(OUTPUT_DIR, f"{file_id}_synthesis.txt")
                        with open(text_path, "w") as f:
                            f.write(final_text)
                        
                        print(f"  Final: {final_text[:100]}...")
                    
                    elif output_type == "audio":
                        audio_tensor = output.multimodal_output["audio"]
                        audio_path_out = os.path.join(OUTPUT_DIR, f"{file_id}_response.wav")
                        sf.write(audio_path_out, audio_tensor.detach().cpu().numpy(), samplerate=24000)
                        print(f"  Audio saved: {audio_path_out}")
            
            trace["Stage"] = "tool_synthesis"
            trace["Status"] = "Success"
        
        else:
            print("  [NO TOOL NEEDED] Direct response")
            trace["Final_Response_Len"] = len(plan_text)
            
            text_path = os.path.join(OUTPUT_DIR, f"{file_id}_direct.txt")
            with open(text_path, "w") as f:
                f.write(plan_text)
            
            trace["Stage"] = "direct"
            trace["Status"] = "Success"
    
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        trace["Status"] = f"Error: {str(e)[:50]}"
    
    finally:
        trace["TotalLatency(s)"] = round(time.time() - t_total_start, 3)
        recorder.log(trace)
    
    return trace["Status"] == "Success"

def run_batch_pipeline():
    """Main entry point - processes all audio files"""
    
    print("=== vLLM-Omni Function Calling Pipeline ===")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recorder = TraceRecorder(TRACE_FILE)
    
    print(f"\n[SCAN] Looking for audio in: {DATASET_DIR}")
    audio_files = []
    for ext in ['*.mp3', '*.flac', '*.wav']:
        audio_files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
    
    audio_files.sort()
    print(f"  Found {len(audio_files)} files")
    
    if not audio_files:
        print("[ERROR] No audio files found!")
        return
    
    print("\n[INIT] Loading vLLM-Omni...")
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
    
    sampling_params_list = [
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.1),
        SamplingParams(temperature=0.9, top_p=0.8, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.05, stop_token_ids=[8294]),
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048, seed=SEED, detokenize=True, repetition_penalty=1.1),
    ]
    
    print(f"\n[PROCESS] Running pipeline on {len(audio_files)} files...")
    print(f"[TRACE] Logging to: {TRACE_FILE}\n")
    
    success_count = 0
    
    try:
        for idx, audio_path in enumerate(audio_files, 1):
            file_id = f"{idx:03d}"
            
            success = process_single_item(
                omni_llm,
                audio_path,
                file_id,
                sampling_params_list,
                recorder
            )
            
            if success:
                success_count += 1
            
            if idx % 10 == 0:
                print(f"\n[PROGRESS] {idx}/{len(audio_files)} completed ({success_count} successful)")
    
    finally:
        print(f"\n{'='*60}")
        print(f"COMPLETED: {success_count}/{len(audio_files)} successful")
        print(f"Traces saved: {TRACE_FILE}")
        print(f"Outputs in: {OUTPUT_DIR}")
        print(f"{'='*60}")
        
        omni_llm.close()

if __name__ == "__main__":
    run_batch_pipeline()