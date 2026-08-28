from __future__ import annotations

import copy
from typing import Any

from ..config import validate_candidate_and_selection
from .deepseek import DeepSeekClient


SYSTEM_PROMPT = r"""
\u4f60\u662f FixedEnzymeAgent \u7684\u201c\u53c2\u6570\u89e3\u6790\u5668\u201d\u3002
\u4f60\u7684\u552f\u4e00\u4efb\u52a1\u662f\u628a\u7528\u6237\u7684\u86cb\u767d\u8d28\u5de5\u7a0b\u81ea\u7136\u8bed\u8a00\u9700\u6c42\u8f6c\u6362\u4e3a\u4e25\u683c JSON\u3002

\u91cd\u8981\u8fb9\u754c\uff1a
1. \u4f60\u53ea\u80fd\u5efa\u8bae candidate \u4e0e selection \u4e0b\u5141\u8bb8\u4fee\u6539\u7684\u767d\u540d\u5355\u53c2\u6570\u3002
2. \u4f60\u7edd\u4e0d\u80fd\u4fee\u6539 sequence\u3001output_dir\u3001\u6a21\u578b\u8def\u5f84\u3001checkpoint\u3001Python \u73af\u5883\u3001GPU\u3001cache\u3001timeout \u6216 API key\u3002
3. activity_proxy \u662f\u6a21\u578b\u76f8\u5bf9\u8bc4\u5206\uff0c\u4e0d\u662f\u5b9e\u9a8c\u6d3b\u6027\u3001kcat\u3001Km \u6216\u7edd\u5bf9\u9176\u6d3b\u3002
4. UniStab \u7684 ddG \u597d\u574f\u65b9\u5411\u7531 stability_direction \u51b3\u5b9a\uff1b\u7528\u6237\u672a\u660e\u786e\u65f6\u8fd4\u56de null\uff0c\u4e0d\u8981\u731c\u3002
5. \u7528\u6237\u6ca1\u6709\u660e\u786e\u7ed9\u51fa\u7684\u53c2\u6570\u8bf7\u8fd4\u56de null\uff0c\u7531\u7a0b\u5e8f\u4fdd\u7559\u539f\u914d\u7f6e\u3002
6. allowed_positions / excluded_positions \u4f7f\u7528 1-based \u6b8b\u57fa\u4f4d\u7f6e\u6574\u6570\u6570\u7ec4\u3002
7. weights \u662f\u504f\u597d\u6743\u91cd\uff0c\u4e0d\u8981\u6c42\u548c\u4e3a 1\uff1b\u7a0b\u5e8f\u4f1a\u5f52\u4e00\u5316\u3002
8. \u5bc6\u7801\u5b50\u4f18\u5316\u4e0e\u4e00\u4e2a\u6216\u591a\u4e2a\u76ee\u6807\u8868\u8fbe\u7269\u79cd\u7531\u7a0b\u5e8f\u72ec\u7acb\u89e3\u6790\uff0c\u4e0d\u5c5e\u4e8e candidate/selection\uff1b\u4e0d\u8981\u628a\u201c\u5bc6\u7801\u5b50\u4f18\u5316\u201d\u3001\u5bbf\u4e3b\u540d\u79f0\u3001\u62c9\u4e01\u5b66\u540d\u6216 TaxID \u6620\u5c04\u6210\u7b5b\u9009\u53c2\u6570\u3002
9. candidate.top_k \u662f\u7a0b\u5e8f\u5185\u90e8\u5019\u9009\u6c60\u5927\u5c0f\uff0c\u7981\u6b62\u6839\u636e\u7528\u6237\u81ea\u7136\u8bed\u8a00\u4fee\u6539\uff0c\u5fc5\u987b\u59cb\u7ec8\u8fd4\u56de null\u3002
10. \u7528\u6237\u8bf4\u201c\u8fd4\u56de N \u4e2a\u201d\u201c\u6700\u7ec8\u8fd4\u56de N \u4e2a\u201d\u201c\u7ed9\u6211 N \u4e2a\u5019\u9009\u201d\u201cTop N\u201d\u201c\u6700\u7ec8\u9009 N \u4e2a\u201d\u7b49\uff0cN \u53ea\u80fd\u5199\u5165 selection.top_k\u3002
11. candidate.top_k \u548c selection.top_k \u542b\u4e49\u5b8c\u5168\u4e0d\u540c\uff1a
    - candidate.top_k\uff1a\u5185\u90e8\u5019\u9009\u6c60\u5927\u5c0f\uff0c\u7531\u7a0b\u5e8f\u914d\u7f6e\u51b3\u5b9a\uff0c\u4e0d\u5141\u8bb8 LLM \u4fee\u6539\u3002
    - selection.top_k\uff1a\u6700\u7ec8\u8fd4\u56de\u7ed9\u7528\u6237\u7684\u5019\u9009\u6570\u91cf\uff0c\u53ef\u4ee5\u6839\u636e\u7528\u6237\u8981\u6c42\u4fee\u6539\u3002
12. \u53ea\u80fd\u8f93\u51fa JSON\uff0c\u4e0d\u8981\u8f93\u51fa Markdown\u3001\u89e3\u91ca\u6bb5\u843d\u6216\u4ee3\u7801\u56f4\u680f\u3002

\u5fc5\u987b\u8f93\u51fa\u5982\u4e0b JSON \u7ed3\u6784\uff1a
{
  "candidate": {
    "top_k": null,
    "allowed_positions": null,
    "excluded_positions": null
  },
  "selection": {
    "target_ph": null,
    "ph_tolerance": null,
    "min_activity_proxy": null,
    "stability_direction": null,
    "stability_operator": null,
    "stability_threshold": null,
    "weights": {
      "ph": null,
      "activity": null,
      "stability": null
    },
    "top_k": null
  },
  "interpretation": "\u7528\u4e00\u53e5\u4e2d\u6587\u6982\u62ec\u4f60\u5982\u4f55\u7406\u89e3\u7528\u6237\u76ee\u6807"
}
""".strip()


def _merge_non_null(
    dst: dict[str, Any],
    src: dict[str, Any],
    keys: list[str],
) -> None:
    for key in keys:
        value = src.get(key)
        if value is not None:
            dst[key] = value


def parse_user_request(
    cfg: dict[str, Any],
    *,
    client: DeepSeekClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = str(cfg.get("user_request") or "").strip()
    if not request:
        raise ValueError("\u5df2\u542f\u7528 LLM \u53c2\u6570\u89e3\u6790\uff0c\u4f46\u6ca1\u6709 user_request/--request")

    llm_cfg = cfg["llm"]

    # candidate.top_k \u53ea\u4f5c\u4e3a\u4e0a\u4e0b\u6587\u544a\u8bc9 LLM \u5f53\u524d\u5185\u90e8\u5019\u9009\u6c60\u5927\u5c0f\uff0c
    # \u4f46 LLM \u65e0\u6743\u4fee\u6539\u5b83\u3002
    current = {
        "candidate": cfg["candidate"],
        "selection": cfg["selection"],
    }

    parsed = client.json_object(
        model=llm_cfg.get("parameter_model", "deepseek-v4-flash"),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "\u7528\u6237\u9700\u6c42\u5982\u4e0b\uff1a\n"
            f"{request}\n\n"
            "\u5f53\u524d\u53c2\u6570\u5982\u4e0b\u3002\n"
            "\u6ce8\u610f\uff1acandidate.top_k \u662f\u5185\u90e8\u5019\u9009\u6c60\u5927\u5c0f\uff0c\u53ea\u4f9b\u53c2\u8003\uff0c\u7981\u6b62\u4fee\u6539\uff1b"
            "\u7528\u6237\u8981\u6c42\u7684\u6700\u7ec8\u5019\u9009\u6570\u91cf\u53ea\u80fd\u5199\u5165 selection.top_k\u3002\n"
            "\u4ec5\u5728\u7528\u6237\u660e\u786e\u8981\u6c42\u65f6\u8986\u76d6\u5176\u4ed6\u5141\u8bb8\u53c2\u6570\u3002\u8bf7\u8f93\u51fa JSON\uff1a\n"
            f"{current}"
        ),
        max_tokens=int(llm_cfg.get("max_tokens", 4000)),
    )

    resolved = copy.deepcopy(cfg)

    candidate_src = parsed.get("candidate") or {}
    selection_src = parsed.get("selection") or {}

    if not isinstance(candidate_src, dict) or not isinstance(selection_src, dict):
        raise ValueError("DeepSeek \u53c2\u6570 JSON \u4e2d candidate/selection \u5fc5\u987b\u662f object")

    # ------------------------------------------------------------
    # \u5f3a\u5236\u4fdd\u62a4 candidate.top_k
    # ------------------------------------------------------------
    # \u5373\u4f7f LLM \u9519\u8bef\u8fd4\u56de candidate.top_k\uff0c\u4e5f\u6c38\u8fdc\u5ffd\u7565\u3002
    candidate_src["top_k"] = None

    # \u540c\u65f6\u4fee\u6539 parsed\uff0c\u4fdd\u8bc1 --print-llm \u8f93\u51fa\u4e0d\u4f1a\u518d\u663e\u793a\u9519\u8bef\u7684
    # candidate.top_k=3\u3002
    if not isinstance(parsed.get("candidate"), dict):
        parsed["candidate"] = {}
    parsed["candidate"]["top_k"] = None

    # candidate \u4e2d\u53ea\u5141\u8bb8 LLM \u4fee\u6539\u4f4d\u7f6e\u7ea6\u675f\u3002
    _merge_non_null(
        resolved["candidate"],
        candidate_src,
        [
            "allowed_positions",
            "excluded_positions",
        ],
    )

    # selection.top_k \u624d\u662f\u7528\u6237\u8981\u6c42\u7684\u201c\u6700\u7ec8\u8fd4\u56de N \u4e2a\u201d\u3002
    _merge_non_null(
        resolved["selection"],
        selection_src,
        [
            "target_ph",
            "ph_tolerance",
            "min_activity_proxy",
            "stability_direction",
            "stability_operator",
            "stability_threshold",
            "top_k",
        ],
    )

    weights_src = selection_src.get("weights") or {}
    if not isinstance(weights_src, dict):
        raise ValueError(
            "DeepSeek \u53c2\u6570 JSON \u4e2d selection.weights \u5fc5\u987b\u662f object"
        )

    for key in ["ph", "activity", "stability"]:
        value = weights_src.get(key)
        if value is not None:
            resolved["selection"]["weights"][key] = float(value)

    # \u6743\u91cd\u5f52\u4e00\u5316\u3002
    weights = resolved["selection"]["weights"]
    total = sum(
        float(weights[k])
        for k in ["ph", "activity", "stability"]
    )

    if total > 0:
        for key in ["ph", "activity", "stability"]:
            weights[key] = float(weights[key]) / total

    # \u6700\u540e\u4ecd\u4f7f\u7528\u786e\u5b9a\u6027\u914d\u7f6e\u6821\u9a8c\u3002
    validate_candidate_and_selection(
        resolved,
        strict_ph_target=True,
    )

    return resolved, parsed