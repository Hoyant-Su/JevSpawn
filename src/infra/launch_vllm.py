import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(config["device"]), OMP_NUM_THREADS=str(config["cpu_threads"]),
        HF_HOME=str(cache / "huggingface"), XDG_CACHE_HOME=str(cache),
        TRITON_CACHE_DIR=str(cache / "triton"), VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"), TMPDIR=str(cache),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), CUDA_CACHE_PATH=str(cache / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"), NUMBA_CACHE_DIR=str(cache / "numba"),
        PYTHONDONTWRITEBYTECODE="1")
    command = [config["python"], "-m", "vllm.entrypoints.openai.api_server",
               "--model", config["model_path"], "--served-model-name", config["model_name"],
               "--host", config["host"], "--port", str(config["port"]),
               "--dtype", config["dtype"], "--tensor-parallel-size", str(config["tensor_parallel_size"]),
               "--max-model-len", str(config["max_model_len"]),
               "--max-num-seqs", str(config["batch_size"]),
               "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
               "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
               "--kv-cache-memory-bytes", str(config["kv_cache_memory_bytes"]),
               "--seed", str(config["seed"]), "--generation-config", "vllm",
               "--enable-prefix-caching" if config["prefix_caching"] else "--no-enable-prefix-caching",
               "--no-enable-chunked-prefill",
               "--limit-mm-per-prompt", '{"image": 0, "video": 0}']
    print(json.dumps(command), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
