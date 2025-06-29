import os
from nanovllm import LLM as NanoLLM, SamplingParams as NanoSamplingParams
from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU, HellaSwag

# Helper class to match deepeval's expected output structure
class MMLUAnswer:
    def __init__(self, answer):
        self.answer = answer


# --- Custom LLM Wrappers ---
class NanoVLLMAdapter(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def get_model_name(self):
        return f"NanoVLLM ({self.model_path})"

    def load_model(self):
        # Instantiate NanoVLLM
        self.model = NanoLLM(self.model_path, enforce_eager=False, max_model_len=4096)
        return self.model

    def generate(self, prompt: str) -> str:
        # Single-example generation
        nanoparams = NanoSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
        outputs = self.model.generate([prompt], nanoparams)
        # Assume outputs is a list of tokens; decode to string as needed
        return outputs[0].outputs[0].text

    async def a_generate(self, prompt: str) -> str:
        # Async wrapper
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str], **kwargs) -> list[list[MMLUAnswer]]:
        nanoparams = NanoSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
        outputs = self.model.generate(prompts, nanoparams)
        return [[MMLUAnswer(output.outputs[0].text)] for output in outputs]

    async def a_batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        return self.batch_generate(prompts, **kwargs)

class VLLMAdapter(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def get_model_name(self):
        return f"vLLM ({self.model_path})"

    def load_model(self):
        # Instantiate vLLM
        self.model = VLLM(model=self.model_path, max_model_len=4096)
        return self.model

    def generate(self, prompt: str) -> str:
        # Single-example generation
        vllm_params = VLLMSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
        out = self.model.generate([prompt], vllm_params)
        # Extract text from first completion
        return out[0].outputs[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str], **kwargs) -> list[list[MMLUAnswer]]:
        vllm_params = VLLMSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
        out = self.model.generate(prompts, vllm_params)
        return [[MMLUAnswer(o.outputs[0].text)] for o in out]

    async def a_batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        return self.batch_generate(prompts, **kwargs)

# --- Benchmark Script ---
def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Initialize adapters
    nano_model = NanoVLLMAdapter(model_path)
    vllm_model = VLLMAdapter(model_path)
    vllm_model.load_model()

    # Choose benchmarks
    benchmarks = [MMLU(), HellaSwag()]

    # Evaluate each model on each benchmark
    for bm in benchmarks:
        print(f"=== Benchmark: {bm.__class__.__name__} ===")

        adapter = vllm_model
        print(f"Evaluating {adapter.get_model_name()}...")
        results = bm.evaluate(model=adapter, batch_size=8)
        print(f"Overall Score: {results.overall_score}\n")
        # Optionally inspect detailed scores
        print(results.task_scores)
        print(results.predictions.head())

if __name__ == "__main__":
    main()
