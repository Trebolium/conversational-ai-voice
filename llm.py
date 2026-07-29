"""Hybrid LLM backend for the conversational voice AI pipeline.

Selection logic:
- If ANTHROPIC_API_KEY is set, use Claude (hosted, higher quality).
- Otherwise, fall back to a local Ollama model (zero-setup, no API key).

This module exposes a single function, generate_response(), so the rest of
the pipeline (ASR -> [this] -> TTS) never needs to know which backend is
active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from metrics import measure

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, friendly voice support assistant. Speak in short, "
    "natural sentences suitable for text-to-speech -- avoid bullet points, "
    "markdown, or long lists."
)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


def active_backend() -> str:
    """Return which backend will be used, based on environment config."""
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"


def generate_response(
    messages: list[Message],
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate a reply given conversation history. Routes to whichever
    backend is active."""
    backend = active_backend()
    with measure(f"llm.generate:{backend}"):
        if backend == "claude":
            return _generate_claude(messages, system)
        return _generate_ollama(messages, system)


def _generate_claude(messages: list[Message], system: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system,
        messages=[{"role": m.role, "content": m.content} for m in messages],
    )
    return response.content[0].text


def _generate_ollama(messages: list[Message], system: str) -> str:
    import ollama

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

    client = ollama.Client(host=host)
    ollama_messages = [{"role": "system", "content": system}]
    ollama_messages += [{"role": m.role, "content": m.content} for m in messages]

    response = client.chat(model=model, messages=ollama_messages)
    return response["message"]["content"]


if __name__ == "__main__":
    # Quick manual test: python -m conversational_ai_voice.llm
    print(f"Active backend: {active_backend()}")
    history = [Message(role="user", content="How do I install ESPnet2 without Kaldi?")]
    print(generate_response(history))
