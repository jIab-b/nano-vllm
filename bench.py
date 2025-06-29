import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import set_attention_backend
#from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    for backend in ["pytorch", "custom"]:
        print(f"--- Benchmarking {backend.upper()} Backend ---")
        set_attention_backend(backend)
        
        # We must re-initialize the LLM to make it re-evaluate the backend setting
        llm = LLM(path, enforce_eager=False, max_model_len=4096)

        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

        # Warm-up run
        llm.generate(["Benchmark: "], SamplingParams())

        # Benchmark run
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        elapsed_time = time.time() - t
        
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / elapsed_time
        
        print(f"Total Tokens: {total_tokens}")
        print(f"Time Taken:   {elapsed_time:.2f}s")
        print(f"Throughput:   {throughput:.2f} tok/s\n")
        
        # Clean up to release memory for the next run
        llm.exit()
        del llm
        # It's good practice to clear the cache, although the process exit should handle it
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
