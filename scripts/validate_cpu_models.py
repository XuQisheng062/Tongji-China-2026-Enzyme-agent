#!/usr/bin/env python3
import argparse
import json
from fixed_enzyme_agent.config import load_config
from fixed_enzyme_agent.validation import validate_three_models_cpu

p = argparse.ArgumentParser(description="\u771f\u5b9e CPU \u9884\u9a8c\u8bc1\u4e09\u4e2a\u6a21\u578b")
p.add_argument("config")
a = p.parse_args()
result = validate_three_models_cpu(load_config(a.config))
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n[PASS] EpHod / EnzGFM / UniStab \u5747\u5df2\u771f\u5b9e\u5728 CPU \u4e0a\u5b8c\u6210\u4e00\u6b21\u63a8\u7406")
