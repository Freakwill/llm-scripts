#!/usr/bin/env python3

"""
Usage:
    ask.py "What is the meaning of life" --model deepseek-v4-flash
Requirements:
    openai
"""

from pathlib import Path

MEMORY_DIR = Path("~/Scripts/memory/").expanduser()  # Configure the path for memory

from openai import OpenAI

from llm_providers import *
from utils import get_api_key


class Client(OpenAI):
    """Wrapper of openai client class
    
    Attributes:
        max_retries (int): max of retries to connect the server
    """
    
    def __init__(self, provider='deepseek'):
        super().__init__(api_key=get_api_key(provider),
            base_url=url_dict[provider])
        self.max_retries=5

    def reply(self, **kwargs):
        for i in range(self.max_retries):
            try:
                return self.chat.completions.create(**kwargs)
            except Exception as e:
                raise e
        raise Exception("retries exhausted")


description_dict = {"default": """回答必须简洁明了，不过度举例，不必重复性表达，如不重复问题，也不用总结。要点多时，可以罗列，少用连词。
默认设置：
- 程序问题只要代码（英文注释）
- 翻译只要一个翻译结果（地道专业的）；如“翻译：我爱你”，“I love you”
""",
"translate": "翻译。只要一个翻译结果（地道专业）；例如，我爱你 -> I love you",
"professional": """回答必须专业，格式规范。内容较多时清晰罗列要点。数学（包括机器学习）问题尽可能用数学公式（常用符号不必解释）；哲学问题要深刻剖析，不流于平庸""",
"math": """尽可能用数学公式（不要过多解释），最后列出经典文献""",
"buddhism": """你是一位以佛陀智慧为内核的智能体。请用佛教教义和智慧启迪我、开导我，回答应平和、慈悲、直指人心，避免冗长说教。""",
"english": "用英文回答我的问题",
"coder": "写出简洁优化的程序。代码简短就不用文字解释。注释一律英文。"
}


def reply(user_input, model="deepseek/deepseek-v4-flash", description="default", memory_tag=None, **kwargs):
    """
    Sends a user query to the DeepSeek chat API and returns a concise, no-frills response.

    Parameters:
        user_input (str): The user's question or prompt.
        model (str): The DeepSeek model to use. Four allowed forms:
           - provider/model(deepseek/deepseek-v4-flash)
           - provider(deepseek): use default model
           - alias: kimi==moonshot, qwen==dashscope
           - model(in form of provider-... e.g. deepseek-v4-flash)
        memory_tag(str): the tag of memory system; None means to not use memory
    """

    def _resolve(model):
        if '/' in model:
            provider, model = model.split('/')
        elif model in default_model:
            if model in url_dict:
                provider = model
                model = default_model[provider]
        elif model in alias:
            provider, model = alias[model].split('/')
        else:
            _provider = model.partition('-')[0]
            provider = model_provider.get(_provider, _provider)
        return provider, model

    # set client
    provider, model = _resolve(model)

    if provider not in url_dict:
        raise Exception(f'Provider `{provider}` is not supported or valid!')

    client = Client(provider)

    # set description
    description = description_dict.get(description, description)

    # set context (seq. of messages)
    messages = [
        {"role": "system", "content": description},
        {"role": "user", "content": user_input}
    ]

    # if memory_tag is True and description in description_dict:
    #     memory_tag = description

    if memory_tag is not None:
        mem_path = MEMORY_DIR / memory_tag
        if mem_path.exists():
            content = mem_path.read_text().strip()
            if content:
                messages.insert(1, {"role": "system", "content": f"The previous dialogue review: {content}"})
        else:
            mem_path.parent.mkdir(parents=True)
            mem_path.touch()

    response = client.reply(model=model, messages=messages, **kwargs)

    if kwargs.get("stream", False):
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                print(chunk_content, end="")
                content += chunk_content
    else:
        content = response.choices[0].message.content
        print(content)
    print()

    if memory_tag is not None:
        response = client.reply(model=model,
            messages=messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"""Save the history of conversation in {mem_path.name}. If it is too long,
                summarize the conversation and save it as a recursive memory in {mem_path.name}.
                Start with 'Content review:' and nothing else. Then provide the content directly. Keep it under 500 words."""}
                ],
            **kwargs)
        summary = response.choices[0].message.content
        mem_path.write_text(summary)


if __name__ == "__main__":
    from fire import Fire
    Fire(reply)
