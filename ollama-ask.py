#!/usr/bin/env python3

"""
Unified Ollama ask script powered by litellm — supports local and cloud Ollama.

Usage:
    ollama-ask.py "What is the meaning of life?"
    ollama-ask.py "Explain quicksort" --model gemma3
    ollama-ask.py "Translate: 你好世界" -d translate
    ollama-ask.py "Write a sort function" -d coder
    ollama-ask.py "Hello" --host cloud --stream
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.suppress_debug_info = True

DEFAULT_MODEL = "gpt-oss:20b"

DESCRIPTION = {
    "default": (
        "Be concise and direct. "
        "Skip preambles, disclaimers, and wrap-up summaries. "
        "No rhetorical questions, no filler phrases (e.g. 'Sure!', 'Of course!', "
        '"Great question!", "I\'d be happy to help"). '
        "For code: give the code with essential comments only, no explanations. "
        "For translations: give only the translation, no commentary. "
        "Use bullet points only when listing distinct items; avoid prose padding. "
        "Answer as if every word costs money."
    ),
    "translate": (
        "You are a professional translator. "
        "Translate the given text accurately and idiomatically. "
        "Output only the translation — no explanations, no alternatives, no notes. "
        "Preserve the original tone and register. "
    ),
    "coder": (
        "You are a senior software engineer. "
        "Write production-quality code. "
        "Output the code directly with minimal, essential inline comments. "
        "Do NOT explain what the code does, do NOT list alternatives, "
        "do NOT describe the approach — just the implementation. "
        "If multiple files are needed, label each with a file-path comment. "
        "Follow language idioms and best practices. "
    ),
    "explain": (
        "You are a patient teacher. "
        "Explain the concept clearly from first principles. "
        "Use analogies where helpful. "
        "Build up from simple to complex. "
        "Highlight key takeaways at the end. "
        "No unnecessary digressions or self-references. "
    ),
    "professional": (
        "You are a domain expert. "
        "Answer with authority and precision. "
        "Structure your response: core answer first, then supporting details. "
        "Use formal, precise language. "
        "Cite sources or standard references when applicable. "
        "No casual tone, no emojis, no fillers. "
    ),
    "creative": (
        "You are a creative writer. "
        "Write with vivid imagery, emotional resonance, and original voice. "
        "Match the requested format (poem, story, dialogue, etc.). "
        "Avoid clichés. Make every sentence earn its place. "
    ),
    "shell": (
        "You are a command-line expert on macOS. "
        "Output shell commands directly — no markdown fences, no explanations. "
        "Use modern alternatives where appropriate (ripgrep > grep, fd > find). "
        "One command per request unless the task requires a pipeline. "
    ),
    "buddhism": (
        "You are a wise teacher grounded in Buddhist philosophy. "
        "Draw from the Dharma — the Four Noble Truths, the Eightfold Path, "
        "emptiness, impermanence, compassion, and mindfulness — to illuminate "
        "the question. "
        "Speak with clarity, warmth, and depth, as if guiding a sincere seeker. "
        "Avoid dogma, proselytizing, or superficial platitudes. "
        "Answer the heart of the question, not just its surface. "
    ),
}

HOST_MAP = {
    "local": "http://localhost:11434",
    "cloud": "https://ollama.com",
}


def _model_installed_locally(model: str) -> bool:
    """Check if *model* is available in the local Ollama instance."""
    try:
        import ollama
    except ImportError:
        return False
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return False
    for name in names:
        base = name.split(":")[0]
        if model == name or model == base:
            return True
    return False


def ask(
    prompt: str = "",
    model: str = DEFAULT_MODEL,
    description: str = "default",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
    host: str = "local",
    newline: bool = True,
) -> None:
    """Send a prompt to an Ollama model via litellm and print the response.

    When *host* is 'local' and the model is not installed, auto-switches to cloud.

    Args:
        prompt: The user prompt.
        model: Ollama model name (e.g. llama3.2, qwen2.5, gpt-oss:120b).
        description: Preset system prompt — 'default', 'translate', 'coder',
            'explain', 'professional', 'creative', 'shell'.
        system: Custom system prompt (overrides *description*).
        temperature: Sampling temperature (0.0 – 2.0).
        max_tokens: Maximum tokens in the response.
        stream: Stream tokens as they arrive.
        host: 'local' (http://localhost:11434) or 'cloud' (https://ollama.com).
        newline: Print trailing newline after response (default True).
    """
    if not prompt:
        raise ValueError("no prompt provided")

    if ":" not in model:
        model = f"{model}:latest"

    if host == "local" and not _model_installed_locally(model):
        host = "cloud"

    os.environ["OLLAMA_API_BASE"] = HOST_MAP.get(host, host)

    if system:
        desc = system
    else:
        desc = DESCRIPTION.get(description, DESCRIPTION["default"])

    messages: list[dict[str, str]] = []
    messages.append({"role": "system", "content": desc})
    messages.append({"role": "user", "content": prompt})

    litellm_model = f"ollama/{model}"

    response = litellm.completion(
        model=litellm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        out: list[str] = []
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                print(text, end="", flush=True)
                out.append(text)
    else:
        text = response.choices[0].message.content
        print(text, end="")
        out = [text]

    if newline and not (out and out[-1].endswith("\n")):
        print()


if __name__ == "__main__":
    from fire import Fire

    Fire(ask)
