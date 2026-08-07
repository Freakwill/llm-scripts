#!/usr/bin/env python3

"""
Providers of LLMs
"""


# provider -> url
url_dict = {"deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "minimax": "https://api.minimaxi.com/v1",
    "modelscope":"https://api-inference.modelscope.cn/v1",
    "wavespeed": "https://llm.wavespeed.ai/v1",
    "gitcode": "https://api-ai.gitcode.com/v1",
    "opencode": "https://opencode.ai/zen/v1"}


# provider -> default model
default_model = {"deepseek": "deepseek/deepseek-v4-flash",
    "moonshot": "moonshot/kimi-k3",
    "qwen": "dashscope/qwen-plus",
    "minimax": "minimax/MiniMax-M3",
    "wavespeed": "anthropic/claude-opus-4.8",
    "gitcode": "zai-org/GLM-5.2",
    "opencode": "opencode/big-pickle"}


# alias of models (use farmiliar words)
alias = {
    "kimi": "moonshot/kimi-k3",
    "qwen": "dashscope/qwen-plus",
    "zen": "opencode/big-pickle"}
