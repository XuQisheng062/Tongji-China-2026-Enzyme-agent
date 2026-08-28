#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os

from fixed_enzyme_agent.llm.deepseek import DeepSeekClient


p = argparse.ArgumentParser(description="DeepSeek API \u547d\u4ee4\u884c\u8fde\u901a\u6027\u6d4b\u8bd5")
p.add_argument("--api-key")
p.add_argument("--ask-api-key", action="store_true")
p.add_argument("--model", default="deepseek-v4-flash", choices=["deepseek-v4-flash", "deepseek-v4-pro"])
p.add_argument("--prompt", default="\u8bf7\u53ea\u56de\u590d\uff1aDeepSeek API OK")
a = p.parse_args()

key = a.api_key or os.environ.get("DEEPSEEK_API_KEY")
if not key and a.ask_api_key:
    key = getpass.getpass("DeepSeek API key: ")

client = DeepSeekClient(api_key=key)
print(client.text(
    model=a.model,
    system_prompt="You are a concise API connectivity tester.",
    user_prompt=a.prompt,
    max_tokens=256,
    thinking=False,
))
