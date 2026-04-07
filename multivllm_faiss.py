#!/usr/bin/env python3

import os
import json
import re
import time
import csv
import glob
import traceback
import faiss
import numpy as np
import soundfile as sf
from typing import Any, Optional, Tuple, Dict, List
from sentence_transformers import SentenceTransformer
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM


SEED = 42
DATASET_DIR = "/dataset_generated"
OUTPUT_DIR = "./outputs"
TRACE_FILE = "./omni_vllm_traces.csv"
CHECKPOINT_FILE = "./checkpoint_completed.json"

SYSTEM_PROMPT = """You are a helpful voice assistant.
If the user asks to look up a customer or provides a phone number, output ONLY this JSON:
{"function_call": {"name": "lookup_customer", "arguments": {"phone_number": "<number>"}}}
If the user asks a general question, just answer them directly and conversationally."""


# Stage-0 (Thinker)  – text generation, normal settings
# Stage-1 (Talker)   – audio token generation; detokenize=False is critical
# Stage-2 (Token2Wav) – waveform diffusion, no special constraints needed

SAMPLING_PARAMS = [
    SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        seed=SEED,
        repetition_penalty=1.1,
    ),
    SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=400,
        seed=SEED,
        repetition_penalty=1.0,
        detokenize=False,  
        prompt_logprobs=0,  
    ),
    SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        seed=SEED,
        repetition_penalty=1.1,
    ),
]

CUSTOMER_DB = [
    {"id": "001", "name": "Esther Lee",  "plan": "Premium",    "balance": "$0.00",   "phone": "+1234567890"},
    {"id": "002", "name": "John Doe",    "plan": "Basic",      "balance": "$15.50",  "phone": "+1987654321"},
    {"id": "003", "name": "Alice Smith", "plan": "Enterprise", "balance": "-$50.00", "phone": "+1122334455"},
]


def build_faiss_index(embedder):
    texts = [f"{c['name']} {c['phone']}" for c in CUSTOMER_DB]
    embeddings = embedder.encode(texts).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[INIT] FAISS index ready with {index.ntotal} records (dim={embeddings.shape[1]})")
    return index


def execute_tool_call(tool_name: str, arguments: Any, embedder, faiss_index) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid arguments format"})

    if tool_name == "lookup_customer":
        search_query = arguments.get("phone_number", "")
        if not search_query:
            return json.dumps({"error": "No search query provided"})

        query_embedding = embedder.encode([search_query]).astype(np.float32)
        distances, indices = faiss_index.search(query_embedding, k=1)

        best_match_idx = indices[0][0]
        distance = distances[0][0]

        if best_match_idx == -1:
            return json.dumps({"error": "Customer not found (empty index)"})
        if distance > 1.5:
            return json.dumps({"error": f"Customer not found (distance={distance:.3f})"})

        matched = CUSTOMER_DB[best_match_idx].copy()
        matched["status"] = "success"
        matched["faiss_distance"] = float(distance)
        return json.dumps(matched)

    return json.dumps({"error": f"Tool '{tool_name}' not found"})


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
        "AudioInput", "Transcription", "LLM_Plan_Latency(s)",
        "Tool_Name", "Tool_Args", "Tool_Exec_Latency(s)",
        "FAISS_Distance", "LLM_Synth_Latency(s)",
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


def load_checkpoint() -> set:
    """Load set of already-completed file basenames."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(completed), f)


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


def check_generate_signature(omni_llm):
    import inspect
    try:
        sig = inspect.signature(omni_llm.generate)
        params = list(sig.parameters.keys())
        print(f"[INFO] OmniLLM.generate() signature params: {params}")
    except Exception as e:
        print(f"[WARN] Could not inspect OmniLLM.generate() signature: {e}")


def run_batch_pipeline():
    print("=== vLLM-Omni + FAISS Pipeline (Batched) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recorder = TraceRecorder(TRACE_FILE)

    completed = load_checkpoint()
    if completed:
        print(f"[CHECKPOINT] Resuming — {len(completed)} files already done, skipping them.")

    print("[INIT] Loading embedding model for FAISS...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = build_faiss_index(embedder)

    # Scan audio files
    print(f"\n[SCAN] Looking for audio in: {DATASET_DIR}")
    audio_files = sorted(
        f for ext in ['*.mp3', '*.flac', '*.wav']
        for f in glob.glob(os.path.join(DATASET_DIR, ext))
    )
    print(f"  Found {len(audio_files)} files")
    if not audio_files:
        return

    print("\n[LOAD] Pre-loading audio...")
    items = []
    skipped = 0
    for idx, path in enumerate(audio_files, 1):
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename in completed:
            skipped += 1
            continue
        result = load_audio(path)
        if result:
            items.append((f"{idx:03d}", path, result[0], result[1]))
        else:
            print(f"  [SKIP] {os.path.basename(path)}")
    print(f"  Loaded {len(items)} valid files ({skipped} skipped from checkpoint)")

    if not items:
        print("All files already processed. Exiting.")
        return

    print("\n[INIT] Loading vLLM-Omni...")
    omni_llm = OmniLLM(
        model="Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        dtype="bfloat16",
        runtime={"devices": [[0], [1], [2]]},
        init_sleep_seconds=10,
        max_model_len=16384,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )

    check_generate_signature(omni_llm)

    t_start = time.time()

    try:
        print(f"\n[PASS 1] Planning batch ({len(items)} prompts)...")
        plan_prompts = [make_initial_prompt(d, sr) for _, _, d, sr in items]

        t_plan = time.time()
        plan_results = omni_llm.generate(plan_prompts, SAMPLING_PARAMS)
        plan_latency = time.time() - t_plan
        print(f"  Planning done in {plan_latency:.1f}s")

        plan_texts = {}
        for stage_out in plan_results:
            if stage_out.final_output_type == "text":
                for out in stage_out.request_output:
                    plan_texts[int(out.request_id)] = out.outputs[0].text

        tool_needed = []
        for req_idx, (file_id, path, audio, sr) in enumerate(items):
            plan_text = plan_texts.get(req_idx, "")
            basename = os.path.splitext(os.path.basename(path))[0]
            tool_call = parse_tool_call(plan_text)

            if tool_call:
                tool_name, tool_args = tool_call
                print(f"  [{file_id}] Tool call: {tool_name}({tool_args})")

                t_tool = time.time()
                tool_result_str = execute_tool_call(tool_name, tool_args, embedder, faiss_index)
                tool_latency = time.time() - t_tool

                try:
                    faiss_dist = json.loads(tool_result_str).get("faiss_distance", "")
                except Exception:
                    faiss_dist = ""

                tool_needed.append((req_idx, file_id, path, audio, sr,
                                    plan_text, tool_name, tool_args,
                                    tool_result_str, tool_latency, faiss_dist))
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
                completed.add(basename)
                save_checkpoint(completed)

        print(f"\n  {len(tool_needed)} files need tool synthesis")
        print(f"  {len(items) - len(tool_needed)} files answered directly")

        if tool_needed:
            print(f"\n[PASS 2] Per-request synthesis ({len(tool_needed)} prompts)...")
            audio_fail_count = 0

            for si, (req_idx, file_id, path, audio, sr,
                     plan_text, tool_name, tool_args,
                     tool_result, tool_latency, faiss_dist) in enumerate(tool_needed):

                basename = os.path.splitext(os.path.basename(path))[0]
                synth_prompt = make_synth_prompt(audio, sr, plan_text, tool_result)

                final_text = ""
                audio_saved = False
                status = "Success"

                t_synth = time.time()
                try:
                    synth_results = omni_llm.generate([synth_prompt], SAMPLING_PARAMS)
                    synth_latency = time.time() - t_synth

                    for stage_out in synth_results:
                        for out in stage_out.request_output:
                            if stage_out.final_output_type == "text":
                                final_text = out.outputs[0].text
                            elif stage_out.final_output_type == "audio":
                                wav = out.multimodal_output["audio"]
                                sf.write(
                                    os.path.join(OUTPUT_DIR, f"{basename}_response.wav"),
                                    wav.detach().cpu().numpy(),
                                    samplerate=24000
                                )
                                audio_saved = True

                    if not audio_saved:
                        status = "TextOnly"

                except Exception as e:
                    synth_latency = time.time() - t_synth
                    audio_fail_count += 1
                    status = f"AudioFailed:{type(e).__name__}"
                    final_text = plan_text
                    print(f"  [WARN] [{file_id}] Stage-2 failed (req {si}): {e}")
                    traceback.print_exc()

                if final_text:
                    with open(os.path.join(OUTPUT_DIR, f"{basename}_synthesis.txt"), "w") as f:
                        f.write(final_text)

                recorder.log({
                    "TraceID": file_id,
                    "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "AudioInput": os.path.basename(path),
                    "LLM_Plan_Latency(s)": round(plan_latency / len(items), 3),
                    "Tool_Name": tool_name,
                    "Tool_Args": json.dumps(tool_args),
                    "Tool_Exec_Latency(s)": round(tool_latency, 3),
                    "FAISS_Distance": faiss_dist,
                    "LLM_Synth_Latency(s)": round(synth_latency, 3),
                    "Transcription": plan_text[:200],
                    "Final_Response_Len": len(final_text),
                    "Stage": "tool_synthesis",
                    "Status": status,
                })

                # Save checkpoint after each file
                completed.add(basename)
                save_checkpoint(completed)

                print(f"  [{file_id}] {si+1}/{len(tool_needed)} done — "
                      f"audio={'yes' if audio_saved else 'NO'} status={status}")

            print(f"\n  Synthesis complete. Audio failures: {audio_fail_count}/{len(tool_needed)}")

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