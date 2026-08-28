#!/usr/bin/env python3
import argparse
import json
from fixed_enzyme_agent.config import load_config
from fixed_enzyme_agent.preflight import gpu_preflight

p = argparse.ArgumentParser(description="\u68c0\u67e5\u4e09\u4e2a\u6a21\u578b GPU \u73af\u5883")
p.add_argument("config")
a = p.parse_args()
result = gpu_preflight(load_config(a.config))
print(json.dumps(result, ensure_ascii=False, indent=2))
if not all(v.get("ok", False) for v in result.values()):
    raise SystemExit(2)
print("\n[PASS] \u4e09\u4e2a\u6a21\u578b\u73af\u5883\u5747\u53ef\u4f7f\u7528\u9009\u5b9a GPU")
