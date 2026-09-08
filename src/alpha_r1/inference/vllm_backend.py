"""Optional vLLM backend for high-throughput batch inference."""

from .prompts import SYSTEM_PROMPT


class VLLMBackend:
    """Same interface as HFBackend, backed by vLLM's offline engine."""

    def __init__(self, model_id: str, temperature: float = 0.0, top_p: float = 0.7,
                 max_tokens: int = 12288, max_model_len: int = 40960,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM backend requires: pip install vllm"
            ) from e

        self.model_id = model_id
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self._tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        return self.generate_batch([prompt], system=system)[0]

    def generate_batch(self, prompts: list[str], system: str = SYSTEM_PROMPT) -> list[str]:
        texts = [
            self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        outputs = self.llm.generate(texts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
