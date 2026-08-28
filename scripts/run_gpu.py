#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os

from fixed_enzyme_agent.config import load_config
from fixed_enzyme_agent.preflight import gpu_preflight
from fixed_enzyme_agent.workflow import FixedEnzymeWorkflow


p = argparse.ArgumentParser(description="GPU \u6b63\u5f0f\u8fd0\u884c\u56fa\u5b9a\u9176\u7a81\u53d8 Agent")
p.add_argument("config")
p.add_argument("--skip-gpu-check", action="store_true")
p.add_argument("--request", help="\u81ea\u7136\u8bed\u8a00\u7b5b\u9009\u76ee\u6807\uff1b\u63d0\u4f9b\u540e\u81ea\u52a8\u542f\u7528 DeepSeek")
p.add_argument("--api-key", help="DeepSeek API key\uff1b\u66f4\u63a8\u8350 --ask-api-key \u6216 DEEPSEEK_API_KEY")
p.add_argument("--ask-api-key", action="store_true", help="\u7ec8\u7aef\u5b89\u5168\u8f93\u5165 API key\uff0c\u4e0d\u56de\u663e")
p.add_argument("--enable-llm", action="store_true")
p.add_argument("--disable-llm", action="store_true")
p.add_argument("--print-llm", action="store_true", help="\u628a DeepSeek \u8fd4\u56de\u540c\u65f6\u6253\u5370\u5230\u7ec8\u7aef")
a = p.parse_args()

cfg = load_config(a.config)
if a.request:
    cfg["user_request"] = a.request
    cfg["llm"]["enabled"] = True
if a.enable_llm:
    cfg["llm"]["enabled"] = True
if a.disable_llm:
    cfg["llm"]["enabled"] = False

api_key = a.api_key or os.environ.get("DEEPSEEK_API_KEY")
if cfg["llm"].get("enabled") and not api_key and a.ask_api_key:
    api_key = getpass.getpass("DeepSeek API key: ")

if not a.skip_gpu_check:
    check = gpu_preflight(cfg)
    if not all(v.get("ok", False) for v in check.values()):
        print(json.dumps(check, ensure_ascii=False, indent=2))
        raise SystemExit("GPU \u9884\u68c0\u5931\u8d25")

result = FixedEnzymeWorkflow(
    cfg,
    force_cpu=False,
    deepseek_api_key=api_key,
    print_llm=a.print_llm,
).run()
print(json.dumps(result, ensure_ascii=False, indent=2))
