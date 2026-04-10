import os
import subprocess 
import time
import socket
import sys

GPU_ID="0"
TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION=1,0.9
MAX_NUM_SEQS,SERVER_PORT,MAX_MODEL_LEN=512,8081,1024
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
DATASET_PATH= "json/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCHMARK_SCRIPT_PATH ="vLlm/vllm/benchmarks/benchmark_serving.py" 

def wait_for_server(port, timeout=2400):
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=5):
                print(f"Server running on {port}")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"Server not ready on port {port}... waiting...")
            time.sleep(3)
    return False

if __name__ == "__main__":
    print(MODEL_NAME)
    env = os.environ.copy()
    
    cuda_home = env.get('CUDA_HOME', '/usr/local/cuda-11.8')
    cuda_lib_path = f"{cuda_home}/lib64"
    
    if 'LD_LIBRARY_PATH' in env:
        if cuda_lib_path not in env['LD_LIBRARY_PATH']:
            env['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = cuda_lib_path
    
    # Set GPU and vLLM specific variables
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["TORCH_COMPILE"] = "0"
    env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Enable debug logging
    
    # Print environment for debugging
    print("=== Environment Check ===")
    print(f"CUDA_HOME: {env.get('CUDA_HOME', 'NOT SET')}")
    print(f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', 'NOT SET')}")
    print(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
    
    # Test if CUDA is accessible
    print("\n=== Testing CUDA Access ===")
    test_result = subprocess.run(
        ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"],
        env=env,
        capture_output=True,
        text=True
    )
    print(test_result.stdout)
    if test_result.stderr:
        print("STDERR:", test_result.stderr)
    print("========================\n")

    server_command = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", str(SERVER_PORT),
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--enforce-eager",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--swap-space", "16",
        "--disable-custom-all-reduce",
    ]

    client_command = [
        "python3", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://127.0.0.1:{SERVER_PORT}",
        "--endpoint", "/v1/completions",
        "--dataset-name", "sharegpt",
        "--model", MODEL_NAME,  
        "--dataset-path", DATASET_PATH,
        "--num-prompts", str(MAX_NUM_SEQS),
        "--sharegpt-output-len", "64",   
    ]

    server_process = None
    try:
        print("Starting vLLM server")
        print(f"command: {' '.join(server_command)}")
        with open("server.log", "w") as server_log:
            server_process = subprocess.Popen(
                server_command, env=env, stdout=server_log, stderr=subprocess.STDOUT
            )

        if not wait_for_server(SERVER_PORT):
            raise RuntimeError("vLLM server failed to start")

        print("Running benchmark client")
        print("command:", " ".join(client_command))
        with open("client.log", "w") as client_log:
            rc=subprocess.call(client_command, env=env, stdout=client_log, stderr=subprocess.STDOUT)
        if rc!=0:
            print(f"Benchmark exited with code {rc}. See client.log")
            try:
                with open("client.log","r") as f:
                    print("\n\n--- client.log ---")
                    print("".join(f.readlines()[-200:]))
            except Exception:
                pass
            if os.environ.get("KEEP_SERVER")=="1":
                print("KEEP_SERVER=1 set. NOT shutting down server for debugging.")
                sys.exit(rc)
            else:
                raise RuntimeError("Benchmark failed... :(")
            

    except Exception as e:
        print(f"ERROR: {e}")
        try:
            with open("server.log", "r") as f:
                print("\n\n--- server.log (last 200 lines) ---")
                print("".join(f.readlines()[-200:]))
        except Exception:
            pass
    finally:
        if server_process and server_process.poll() is None:
            print("\nShutting down the server....")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server did not terminate, force killing....")
                server_process.kill()
            print("Server shut down. Bye.")