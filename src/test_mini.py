import os
import traceback

import numpy as np
import soundfile as sf
import torch

from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

SEED = 42

def main() -> None:
    print("=== Starting vLLM-Omni Test ===")
    print(f"Environment: VLLM_USE_V1={os.environ.get('VLLM_USE_V1', 'NOT SET')}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}")

    audio_path = "./dataset/sample.wav"

    if not os.path.exists(audio_path):
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        sr = 16000
        sf.write(audio_path, np.zeros(sr, dtype=np.float32), sr)
        print(f"Created dummy audio at {audio_path}")

    print("\n=== Initializing OmniLLM ===")

    device_mapping = [[0], [1], [1]]
    print(f"Using device mapping: {device_mapping}")

    omni_llm = OmniLLM(
        model="Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        dtype="bfloat16",
        runtime={"devices": device_mapping},
        init_sleep_seconds=10,
        max_model_len=2048,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print(f"Loading audio from: {audio_path}")
    audio_data, sample_rate = sf.read(audio_path)

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    audio_data = audio_data.astype(np.float32).copy()
    print(f"Audio shape: {audio_data.shape}, Writable: {audio_data.flags.writeable}, SR: {sample_rate}")

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

    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     max_tokens=512,
    #     top_p=0.95,
    #     top_k=50,
    # )

    # sampling_params_list = [sampling_params, sampling_params, sampling_params]

    thinker_sampling_params = SamplingParams(
        temperature=0.0,  
        top_p=1.0, 
        top_k=-1, 
        max_tokens=2048,
        seed=SEED, 
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.8,
        top_k=40,
        max_tokens=2048,
        seed=SEED, 
        detokenize=True,
        repetition_penalty=1.05,
        stop_token_ids=[8294],
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,  
        top_p=1.0,  
        top_k=-1,  
        max_tokens=2048,
        seed=SEED,  
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]    

    print("\n=== Generating Response ===")
    try:
        prompts = [prompt]
        results = omni_llm.generate(prompts, sampling_params_list)
        
        output_dir = './outputs'
        os.makedirs(output_dir, exist_ok=True)

        for stage_outputs in results:
            if stage_outputs.final_output_type == "text":
                for output in stage_outputs.request_output:
                    request_id = output.request_id
                    text_output = output.outputs[0].text
                    # Save aligned text file per request
                    prompt_text = output.prompt
                    out_txt = os.path.join(output_dir, f"{request_id}.txt")
                    lines = []
                    lines.append("Prompt:\n")
                    lines.append(str(prompt_text) + "\n")
                    lines.append("vllm_text_output:\n")
                    lines.append(str(text_output).strip() + "\n")
                    try:
                        with open(out_txt, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                    except Exception as e:
                        print(f"[Warn] Failed writing text file {out_txt}: {e}")
                    print(f"Request ID: {request_id}, Text saved to {out_txt}")

            elif stage_outputs.final_output_type == "audio":
                for output in stage_outputs.request_output:
                    request_id = output.request_id
                    audio_tensor = output.multimodal_output["audio"]
                    output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
                    sf.write(output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
                    print(f"Request ID: {request_id}, Saved audio to {output_wav}")

        omni_llm.close()

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== Shutting down ===") 
        omni_llm.close()

if __name__ == "__main__":
    main()