import os
import time
from random import randint, seed

# Import from both libraries
from nanovllm import LLM as NanoLLM, SamplingParams as NanoSamplingParams
from vllm import SamplingParams as VLLMSamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine


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
    print(f"NanoVLLM -> Total: {total_tokens}tok, Time: {t_nano:.2f}s, Throughput: {throughput_nano:.2f}tok/s\n")
    del llm_nano


    # --- vLLM Benchmark ---
    print("--- Benchmarking vllm ---")
    engine_args = EngineArgs(model=path, max_model_len=4096)
    llm_vllm_engine = LLMEngine.from_engine_args(engine_args)
    
    # Warm-up
    llm_vllm_engine.add_request("warmup", prompt_token_ids=[1, 2, 3], sampling_params=VLLMSamplingParams())
    while llm_vllm_engine.has_unfinished_requests():
        llm_vllm_engine.step()

    # Timed run
    t_vllm = time.time()
    
    # Add all requests to the engine
    for i in range(num_seqs):
        prompt = prompt_token_ids_base[i]
        sampling_params = VLLMSamplingParams(**sampling_params_args[i])
        request_id = str(i)
        llm_vllm_engine.add_request(request_id, prompt_token_ids=prompt, sampling_params=sampling_params)

    # Run the engine until all requests are finished
    outputs = []
    while llm_vllm_engine.has_unfinished_requests():
        request_outputs = llm_vllm_engine.step()
        for output in request_outputs:
            if output.finished:
                outputs.append(output)
    
    t_vllm = (time.time() - t_vllm)

    throughput_vllm = total_tokens / t_vllm
    print(f"vLLM -> Total: {total_tokens}tok, Time: {t_vllm:.2f}s, Throughput: {throughput_vllm:.2f}tok/s")
    del llm_vllm_engine


if __name__ == "__main__":
    main()