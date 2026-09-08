"""Async OpenRouter client (OpenAI-compatible API)."""

import asyncio
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():  # python-dotenv is optional
        return False

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Thin async wrapper around the OpenAI SDK pointed at OpenRouter.

    The model id is deliberately NOT defaulted: set ``model`` in
    ``configs/generation.yaml`` or pass ``--model`` on the command line.
    """

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096,
                 max_retries: int = 5, retry_delay: float = 2.0,
                 base_url: str = OPENROUTER_BASE_URL):
        if not model or not model.strip():
            raise ValueError(
                "OpenRouter model is not configured. Set `model` in "
                "configs/generation.yaml or pass --model, e.g. "
                '"anthropic/claude-3.7-sonnet".'
            )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment; see .env.example"
            )
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai is required for generation: pip install openai") from e
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                                  timeout=300.0, max_retries=0)

    async def chat(self, system: str, user: str) -> str:
        """Single-turn chat with exponential-backoff retries."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                raise ValueError("empty response")
            except Exception as e:
                if attempt >= self.max_retries - 1:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                print(f"[openrouter] attempt {attempt + 1} failed: {e}; retry in {wait:.0f}s")
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")
