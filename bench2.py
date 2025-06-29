import os
import time
from random import randint, seed

# Import from both libraries
from nanovllm import LLM as NanoLLM, SamplingParams as NanoSamplingParams
from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams


def main():
    seed(0)
    num_seqs = 64
    max_input_len = 1024
    max_output_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # --- Common Data Generation ---
    prompt_token_ids_base = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params_args = [
        dict(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]
    total_tokens = sum(args['max_tokens'] for args in sampling_params_args)


    # --- NanoVLLM Benchmark ---
    print("--- Benchmarking nanovllm ---")
    llm_nano = NanoLLM(path, enforce_eager=False, max_model_len=4096)
    sampling_params_nano = [NanoSamplingParams(**args) for args in sampling_params_args]

    # Warm-up
    llm_nano.generate(["Benchmark: "], NanoSamplingParams())
    
    # Timed run
    t_nano = time.time()
    llm_nano.generate(prompt_token_ids_base, sampling_params_nano, use_tqdm=False)
    t_nano = (time.time() - t_nano)
    
    throughput_nano = total_tokens / t_nano
    del llm_nano


    # --- vLLM Benchmark ---
    print("--- Benchmarking vllm ---")
    llm_vllm = VLLM(model=path, max_model_len=4096)
    sampling_params_vllm = [VLLMSamplingParams(**args) for args in sampling_params_args]
    prompts_vllm = [dict(prompt_token_ids=p) for p in prompt_token_ids_base]

    # Warm-up
    llm_vllm.generate(["Benchmark: "], VLLMSamplingParams())
    
    # Timed run
    t_vllm = time.time()
    llm_vllm.generate(prompts_vllm, sampling_params_vllm, use_tqdm=False)
    t_vllm = (time.time() - t_vllm)
    
    throughput_vllm = total_tokens / t_vllm
    del llm_vllm

    print(f"\nNanoVLLM -> Total: {total_tokens}tok, Time: {t_nano:.2f}s, Throughput: {throughput_nano:.2f}tok/s")
    print(f"vLLM -> Total: {total_tokens}tok, Time: {t_vllm:.2f}s, Throughput: {throughput_vllm:.2f}tok/s")


if __name__ == "__main__":
    main()