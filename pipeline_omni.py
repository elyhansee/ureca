#!/usr/bin/env python3
"""
Internal Reasoning Branch Tracker - GPT-OSS Edition with Resource Tracking
Captures reasoning traces AND resource consumption metrics
"""
import json
import requests
import time
import os
import csv
import glob
import re
import subprocess
from typing import Dict, Any, List

# --- CONFIGURATION ---
DATASET_DIR = "dataset_generated"
WHISPER_URL = "http://127.0.0.1:8000"
GPT_URL = "http://127.0.0.1:8001"
WHISPER_MODEL = "whisper-large-v3"
GPT_MODEL = "gpt-oss"

REASONING_TRACE_FILE = "reasoning_branches_detailed.json"
REASONING_CSV_FILE = "reasoning_analysis.csv"
RESOURCE_CSV_FILE = "resource_consumption.csv"

REASONING_EFFORT = "high"  

# --- TOOLS DEFINITION ---
tools = [{
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

class ResourceMonitor:
    """Monitor GPU and system resources"""
    
    @staticmethod
    def get_gpu_memory():
        """Get current GPU memory usage in MB"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except:
            pass
        return None
    
    @staticmethod
    def get_gpu_utilization():
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except:
            pass
        return None

class ReasoningBranchAnalyzer:
    def __init__(self):
        self.reasoning_traces = []
        self.resource_monitor = ResourceMonitor()
        
    def extract_think_tokens(self, content: str) -> tuple[str, str]:
        """Extract reasoning from <think>...</think> tokens"""
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, content, re.DOTALL)
        
        if matches:
            thinking = '\n'.join(matches)
            final_response = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
            return thinking, final_response
        
        return "", content
    
    def analyze_thinking_process(self, thinking_content: str) -> Dict[str, Any]:
        """Analyze the reasoning branches within <think> content"""
        branches = {
            "raw_thinking": thinking_content,
            "thinking_token_count": len(thinking_content.split()),
            "reasoning_steps": [],
            "decision_points": [],
            "branch_types": [],
            "total_branches": 0
        }
        
        if not thinking_content:
            return branches
        
        lines = thinking_content.split('\n')
        current_step = []
        
        step_markers = [
            r'^step \d+', r'^first,?', r'^second,?', r'^third,?', r'^next,?',
            r'^then,?', r'^now,?', r'^finally,?', r'^\d+\.', r'^\d+\)'
        ]
        
        decision_markers = [
            'should i', 'could i', 'option', 'alternative', 'consider',
            'decide', 'choice', 'if .* then', 'either .* or', 'whether'
        ]
        
        tool_reasoning = [
            'need to use', 'should use', 'call the', 'use the tool',
            'lookup_customer', 'phone number', 'tool call'
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_step:
                    branches['reasoning_steps'].append(' '.join(current_step))
                    current_step = []
                continue
            
            is_step = any(re.match(pattern, line.lower()) for pattern in step_markers)
            if is_step:
                if current_step:
                    branches['reasoning_steps'].append(' '.join(current_step))
                branches['total_branches'] += 1
                branches['branch_types'].append('SEQUENTIAL_STEP')
                current_step = [line]
                continue
            
            is_decision = any(marker in line.lower() for marker in decision_markers)
            if is_decision:
                branches['decision_points'].append({
                    "type": "DECISION_FORK",
                    "line_number": i + 1,
                    "content": line
                })
                branches['branch_types'].append('DECISION_POINT')
                branches['total_branches'] += 1
            
            is_tool_reasoning = any(marker in line.lower() for marker in tool_reasoning)
            if is_tool_reasoning:
                branches['decision_points'].append({
                    "type": "TOOL_REASONING",
                    "line_number": i + 1,
                    "content": line
                })
                branches['branch_types'].append('TOOL_SELECTION')
            
            current_step.append(line)
        
        if current_step:
            branches['reasoning_steps'].append(' '.join(current_step))
        
        if branches['total_branches'] == 0:
            paragraphs = [p.strip() for p in thinking_content.split('\n\n') if p.strip()]
            branches['total_branches'] = len(paragraphs)
            branches['reasoning_steps'] = paragraphs
            branches['branch_types'] = ['IMPLICIT_STEP'] * len(paragraphs)
        
        return branches
    
    def analyze_llm_response(self, response_data: dict, phase: str, user_input: str) -> dict:
        """Main analysis function for gpt-oss responses"""
        reasoning = {
            "phase": phase,
            "timestamp": time.time(),
            "user_input": user_input,
            "branches": {
                "total_branches": 0,
                "thinking_token_count": 0,
                "reasoning_steps": [],
                "decision_points": [],
                "branch_types": [],
                "raw_thinking": None
            },
            "final_decision": "NO_RESPONSE",
            "final_response": None
        }
        
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            return reasoning
        
        choice = response_data['choices'][0]
        message = choice.get('message', {})
        content = message.get('content', '')
        
        thinking, final_response = self.extract_think_tokens(content)
        
        if thinking:
            reasoning['branches'] = self.analyze_thinking_process(thinking)
            print(f"  [THINKING] Found {reasoning['branches']['thinking_token_count']} thinking tokens")
            print(f"  [BRANCHES] Detected {reasoning['branches']['total_branches']} reasoning branches")
        else:
            print(f"  [WARNING] No <think> tokens found in response")
            reasoning['branches'] = self.analyze_thinking_process(content)
        
        reasoning['final_response'] = final_response
        
        tool_calls = message.get('tool_calls', [])
        if tool_calls:
            reasoning['final_decision'] = f"TOOL_USE: {tool_calls[0]['function']['name']}"
        elif final_response.strip().startswith('{'):
            reasoning['final_decision'] = "LAZY_JSON"
        elif final_response:
            reasoning['final_decision'] = "TEXT_RESPONSE"
        
        return reasoning
    
    def generate_summary_csv(self, filename: str):
        """Generate detailed CSV with reasoning metrics"""
        if not self.reasoning_traces:
            return
        
        headers = [
            "TraceID", "Audio_File", "Transcription", 
            "Thinking_Tokens", "Total_Branches", "Decision_Points", 
            "Branch_Types", "Final_Decision", "Status"
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for trace in self.reasoning_traces:
                branches = trace.get('reasoning', {}).get('branches', {})
                
                writer.writerow({
                    "TraceID": trace['trace_id'],
                    "Audio_File": os.path.basename(trace['file_path']),
                    "Transcription": trace.get('transcription', '')[:50],
                    "Thinking_Tokens": branches.get('thinking_token_count', 0),
                    "Total_Branches": branches.get('total_branches', 0),
                    "Decision_Points": len(branches.get('decision_points', [])),
                    "Branch_Types": ', '.join(set(branches.get('branch_types', []))),
                    "Final_Decision": trace.get('reasoning', {}).get('final_decision', 'UNKNOWN'),
                    "Status": "Success" if trace.get('transcription') else "Empty Audio"
                })
    
    def generate_resource_csv(self, filename: str):
        """Generate CSV with resource consumption metrics"""
        if not self.reasoning_traces:
            return
        
        headers = [
            "TraceID", "Audio_File", "Transcription_Time_s", "LLM_Inference_Time_s",
            "Total_Time_s", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens",
            "Thinking_Tokens", "Total_Branches", "Decision_Points",
            "GPU_Memory_Before_MB", "GPU_Memory_After_MB", "GPU_Memory_Delta_MB",
            "GPU_Util_Peak_Percent", "Tokens_Per_Second", "Time_Per_Branch_ms"
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for trace in self.reasoning_traces:
                resources = trace.get('resources', {})
                branches = trace.get('reasoning', {}).get('branches', {})
                
                # Calculate derived metrics
                total_time = resources.get('transcription_time', 0) + resources.get('llm_inference_time', 0)
                total_tokens = resources.get('prompt_tokens', 0) + resources.get('completion_tokens', 0)
                
                tokens_per_sec = 0
                if resources.get('llm_inference_time', 0) > 0:
                    tokens_per_sec = total_tokens / resources.get('llm_inference_time')
                
                time_per_branch = 0
                if branches.get('total_branches', 0) > 0:
                    time_per_branch = (resources.get('llm_inference_time', 0) * 1000) / branches.get('total_branches')
                
                writer.writerow({
                    "TraceID": trace['trace_id'],
                    "Audio_File": os.path.basename(trace['file_path']),
                    "Transcription_Time_s": f"{resources.get('transcription_time', 0):.3f}",
                    "LLM_Inference_Time_s": f"{resources.get('llm_inference_time', 0):.3f}",
                    "Total_Time_s": f"{total_time:.3f}",
                    "Prompt_Tokens": resources.get('prompt_tokens', 0),
                    "Completion_Tokens": resources.get('completion_tokens', 0),
                    "Total_Tokens": total_tokens,
                    "Thinking_Tokens": branches.get('thinking_token_count', 0),
                    "Total_Branches": branches.get('total_branches', 0),
                    "Decision_Points": len(branches.get('decision_points', [])),
                    "GPU_Memory_Before_MB": resources.get('gpu_memory_before', 'N/A'),
                    "GPU_Memory_After_MB": resources.get('gpu_memory_after', 'N/A'),
                    "GPU_Memory_Delta_MB": resources.get('gpu_memory_delta', 'N/A'),
                    "GPU_Util_Peak_Percent": resources.get('gpu_util_peak', 'N/A'),
                    "Tokens_Per_Second": f"{tokens_per_sec:.1f}",
                    "Time_Per_Branch_ms": f"{time_per_branch:.2f}"
                })

def build_manifest_from_directory(directory):
    """Scans the directory for audio files"""
    print(f"\n[SCAN] Looking for audio files in: {directory}")
    
    if not os.path.exists(directory):
        print(f"  [ERROR] Directory not found: {directory}")
        return []

    files = []
    for ext in ['*.mp3', '*.flac', '*.wav']:
        found = glob.glob(os.path.join(directory, ext))
        files.extend(found)
    
    files.sort()
    
    dataset = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            fid = fname.split('_')[0]
        except:
            fid = fname
            
        dataset.append({"id": fid, "file": fpath})
        
    print(f"  [OK] Found {len(dataset)} files.")
    return dataset

def process_item(item: Dict[str, Any], index: int, total: int, analyzer: ReasoningBranchAnalyzer) -> bool:
    trace_record = {
        "trace_id": item['id'],
        "file_path": item['file'],
        "resources": {}
    }
    
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] PROCESSING FILE: {os.path.basename(item['file'])}")
    
    if not os.path.exists(item['file']):
        print(f"  [X] File missing: {item['file']}")
        return False
        
    file_size = os.path.getsize(item['file'])
    print(f"  File Size: {file_size} bytes")

    # 1. TRANSCRIPTION with timing
    print("  [1] Transcribing...")
    t_start = time.time()
    
    try:
        mime = "audio/mpeg" if item['file'].endswith(".mp3") else "audio/flac"
        if item['file'].endswith(".wav"): 
            mime = "audio/wav"
        
        with open(item['file'], "rb") as f:
            files = {"file": (os.path.basename(item['file']), f, mime)}
            r = requests.post(f"{WHISPER_URL}/v1/audio/transcriptions", 
                            files=files, data={"model": WHISPER_MODEL}, timeout=60)
            
            if r.status_code != 200:
                print(f"  [X] Whisper Error {r.status_code}: {r.text}")
                return False
                
            transcription = r.json().get("text", "").strip()
            trace_record['transcription'] = transcription
            
            # Record transcription time
            trace_record['resources']['transcription_time'] = time.time() - t_start
            
            print(f"  Result: \"{transcription}\"")
            print(f"  Time: {trace_record['resources']['transcription_time']:.3f}s")
            
            if not transcription:
                print("  [SKIP] Transcription is empty.")
                analyzer.reasoning_traces.append(trace_record)
                return False

    except Exception as e:
        print(f"  [X] Whisper Exception: {e}")
        return False

    # 2. LLM REASONING with resource tracking
    print("  [2] Analyzing Reasoning Branches (with <think> tokens)...")
    
    # Measure GPU before inference
    gpu_mem_before = analyzer.resource_monitor.get_gpu_memory()
    if gpu_mem_before:
        trace_record['resources']['gpu_memory_before'] = gpu_mem_before
        print(f"  GPU Memory Before: {gpu_mem_before:.1f} MB")
    
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant. Use lookup_customer tool if a phone number is provided.

Reasoning effort: {REASONING_EFFORT}

Think step-by-step about how to handle this request."""
    }
    
    messages = [
        system_message,
        {"role": "user", "content": transcription}
    ]
    
    try:
        payload = {
            "model": GPT_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 4096,
        }
        
        # Start inference timer
        t_inference_start = time.time()
        
        # Sample GPU utilization during inference
        gpu_utils = []
        
        r = requests.post(f"{GPT_URL}/v1/chat/completions", json=payload, timeout=120)
        
        # Record inference time
        inference_time = time.time() - t_inference_start
        trace_record['resources']['llm_inference_time'] = inference_time
        
        # Measure GPU after inference
        gpu_mem_after = analyzer.resource_monitor.get_gpu_memory()
        gpu_util = analyzer.resource_monitor.get_gpu_utilization()
        
        if gpu_mem_after:
            trace_record['resources']['gpu_memory_after'] = gpu_mem_after
            if gpu_mem_before:
                trace_record['resources']['gpu_memory_delta'] = gpu_mem_after - gpu_mem_before
        
        if gpu_util:
            trace_record['resources']['gpu_util_peak'] = gpu_util
        
        if r.status_code != 200:
            print(f"  [X] GPT Error {r.status_code}: {r.text}")
            return False
            
        response = r.json()
        
        # Extract token usage
        usage = response.get('usage', {})
        trace_record['resources']['prompt_tokens'] = usage.get('prompt_tokens', 0)
        trace_record['resources']['completion_tokens'] = usage.get('completion_tokens', 0)
        trace_record['resources']['total_tokens'] = usage.get('total_tokens', 0)
        
        # Debug: Check raw response
        if 'choices' in response and len(response['choices']) > 0:
            raw_content = response['choices'][0]['message'].get('content', '')
            if '<think>' in raw_content:
                print(f"  [✓] Thinking tokens detected in response")
            else:
                print(f"  [!] No thinking tokens found - model may not be using extended reasoning")
        
        analysis = analyzer.analyze_llm_response(response, "PLANNING", transcription)
        trace_record['reasoning'] = analysis
        
        # Log reasoning metrics
        branches = analysis['branches']
        print(f"  Thinking Tokens: {branches['thinking_token_count']}")
        print(f"  Reasoning Branches: {branches['total_branches']}")
        print(f"  Decision Points: {len(branches['decision_points'])}")
        print(f"  Final Decision: {analysis['final_decision']}")
        
        # Log resource metrics
        print(f"  Inference Time: {inference_time:.3f}s")
        print(f"  Token Usage: {trace_record['resources']['total_tokens']} tokens " +
              f"({trace_record['resources']['prompt_tokens']} prompt + " +
              f"{trace_record['resources']['completion_tokens']} completion)")
        
        if gpu_mem_after and gpu_mem_before:
            print(f"  GPU Memory: {gpu_mem_before:.1f} → {gpu_mem_after:.1f} MB " +
                  f"(Δ {trace_record['resources']['gpu_memory_delta']:.1f} MB)")
        
        if branches['total_branches'] > 0:
            time_per_branch = (inference_time * 1000) / branches['total_branches']
            print(f"  Time per Branch: {time_per_branch:.2f} ms")
        
        # Save trace
        analyzer.reasoning_traces.append(trace_record)
        return True

    except Exception as e:
        print(f"  [X] GPT Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_pipeline():
    dataset = build_manifest_from_directory(DATASET_DIR)
    
    if not dataset:
        print("[X] No audio files found to process.")
        return

    analyzer = ReasoningBranchAnalyzer()
    
    print("\n" + "="*60)
    print("GPT-OSS Reasoning Branch Analysis Pipeline")
    print(f"Reasoning Effort: {REASONING_EFFORT.upper()}")
    print("="*60)
    
    success_count = 0
    
    for i, item in enumerate(dataset, 1):
        if process_item(item, i, len(dataset), analyzer):
            success_count += 1
            
    # 3. Save Results
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"Success: {success_count}/{len(dataset)}")
    print(f"{'='*60}")
    
    # Save detailed traces
    with open(REASONING_TRACE_FILE, 'w', encoding='utf-8') as f:
        json.dump(analyzer.reasoning_traces, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed traces saved to: {REASONING_TRACE_FILE}")
    
    # Save summary CSV
    analyzer.generate_summary_csv(REASONING_CSV_FILE)
    print(f"Summary saved to: {REASONING_CSV_FILE}")
    
    # Save resource consumption CSV
    analyzer.generate_resource_csv(RESOURCE_CSV_FILE)
    print(f"Resource consumption saved to: {RESOURCE_CSV_FILE}")
    
    # Print statistics
    print("\n" + "="*60)
    print("REASONING STATISTICS")
    print("="*60)
    
    valid_traces = [t for t in analyzer.reasoning_traces if t.get('reasoning')]
    
    total_thinking_tokens = sum(
        t.get('reasoning', {}).get('branches', {}).get('thinking_token_count', 0) 
        for t in valid_traces
    )
    total_branches = sum(
        t.get('reasoning', {}).get('branches', {}).get('total_branches', 0) 
        for t in valid_traces
    )
    total_decisions = sum(
        len(t.get('reasoning', {}).get('branches', {}).get('decision_points', [])) 
        for t in valid_traces
    )
    
    if valid_traces:
        print(f"Total Thinking Tokens: {total_thinking_tokens:,}")
        print(f"Total Reasoning Branches: {total_branches}")
        print(f"Total Decision Points: {total_decisions}")
        print(f"Average Thinking Tokens per Query: {total_thinking_tokens/len(valid_traces):.1f}")
        print(f"Average Branches per Query: {total_branches/len(valid_traces):.1f}")
        print(f"Average Decisions per Query: {total_decisions/len(valid_traces):.1f}")
    
    print("\n" + "="*60)
    print("RESOURCE CONSUMPTION STATISTICS")
    print("="*60)
    
    # Calculate resource statistics
    total_transcription_time = sum(
        t.get('resources', {}).get('transcription_time', 0) for t in valid_traces
    )
    total_inference_time = sum(
        t.get('resources', {}).get('llm_inference_time', 0) for t in valid_traces
    )
    total_tokens = sum(
        t.get('resources', {}).get('total_tokens', 0) for t in valid_traces
    )
    
    if valid_traces:
        print(f"Total Transcription Time: {total_transcription_time:.1f}s")
        print(f"Total LLM Inference Time: {total_inference_time:.1f}s")
        print(f"Total Processing Time: {total_transcription_time + total_inference_time:.1f}s")
        print(f"Average Transcription Time: {total_transcription_time/len(valid_traces):.3f}s")
        print(f"Average Inference Time: {total_inference_time/len(valid_traces):.3f}s")
        print(f"Total Tokens Processed: {total_tokens:,}")
        print(f"Average Tokens per Query: {total_tokens/len(valid_traces):.1f}")
        
        if total_inference_time > 0:
            print(f"Overall Tokens/Second: {total_tokens/total_inference_time:.1f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    run_pipeline()
 