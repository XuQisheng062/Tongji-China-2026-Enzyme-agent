from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .deepseek import DeepSeekClient


_SYSTEM_PROMPTS = {
    "zh": r"""
\u4f60\u662f FixedEnzymeAgent \u7684\u7ed3\u679c\u5206\u6790\u6a21\u5757\u3002\u8bf7\u53ea\u6839\u636e\u7528\u6237\u7ed9\u51fa\u7684\u7ed3\u6784\u5316\u9884\u6d4b\u548c\u786e\u5b9a\u6027\u70b9\u7a81\u53d8\u63cf\u8ff0\u8fdb\u884c\u5206\u6790\uff0c\u4e0d\u5f97\u53d1\u660e\u4efb\u4f55\u6570\u503c\u3002

\u6a21\u578b\u4e0e\u63cf\u8ff0\u7b26\u542b\u4e49\uff1a
- EnzGFM: activity_proxy\uff0c\u53ea\u80fd\u79f0\u4e3a\u201c\u6d3b\u6027\u4ee3\u7406\u8bc4\u5206/\u76f8\u5bf9\u7a81\u53d8\u8bc4\u5206\u201d\uff0c\u4e0d\u80fd\u79f0\u4e3a\u5b9e\u9a8c\u6d3b\u6027\u3001kcat \u6216\u7edd\u5bf9\u9176\u6d3b\u3002
- EpHod: predicted optimum pH\uff0c\u662f\u8ba1\u7b97\u9884\u6d4b\u3002
- UniStab: predicted ddG\uff0c\u662f\u8ba1\u7b97\u9884\u6d4b\uff1b\u597d\u574f\u65b9\u5411\u5fc5\u987b\u4f9d\u636e stability_direction\u3002
- BLOSUM62\u3001\u758f\u6c34\u6027/\u4f53\u79ef/\u7535\u8377\u53d8\u5316\uff1a\u4ec5\u63cf\u8ff0\u6c28\u57fa\u9178\u66ff\u6362\u672c\u8eab\uff0c\u4e0d\u80fd\u5355\u72ec\u8bc1\u660e\u7a81\u53d8\u6709\u76ca\u6216\u6709\u5bb3\u3002
- pareto_optimal\uff1a\u8868\u793a\u8be5\u5019\u9009\u5728\u5f52\u4e00\u5316 activity/pH/stability \u4e09\u76ee\u6807\u4e0a\u672a\u88ab\u53e6\u4e00\u4e2a\u5408\u683c\u5019\u9009\u540c\u65f6\u652f\u914d\u3002

\u5fc5\u987b\u9075\u5b88\uff1a
1. \u4e0d\u628a\u9884\u6d4b\u63cf\u8ff0\u6210\u5b9e\u9a8c\u9a8c\u8bc1\u7ed3\u679c\u3002
2. \u4e0d\u4fee\u6539\u3001\u8865\u9020\u6216\u53cd\u63a8\u8f93\u5165\u91cc\u6ca1\u6709\u7684\u6570\u636e\u3002
3. \u5206\u6790 activity_proxy\u3001pH\u3001ddG\u3001BLOSUM62\u3001\u7406\u5316\u6027\u8d28\u53d8\u5316\u548c Pareto \u72b6\u6001\u4e4b\u95f4\u7684\u4e92\u8865\u4fe1\u606f\u4e0e trade-off\u3002
4. \u89e3\u91ca final_score \u53ea\u662f\u5f53\u524d\u7528\u6237\u6743\u91cd\u4e0b\u7684\u7efc\u5408\u6392\u5e8f\uff0c\u4e0d\u662f\u751f\u7269\u5b66\u771f\u503c\u3002
5. \u63a8\u8350\u5b9e\u9a8c\u9a8c\u8bc1\u4f18\u5148\u7ea7\uff0c\u5e76\u8bf4\u660e\u6700\u7ec8\u5019\u9009\u4ecd\u9700\u6e7f\u5b9e\u9a8c\u9a8c\u8bc1\u3002
6. \u4f7f\u7528\u4e2d\u6587\u7eaf\u6587\u672c\uff1b\u4e0d\u8981\u8f93\u51fa Markdown \u4ee3\u7801\u56f4\u680f\u3002

\u5efa\u8bae\u7ed3\u6784\uff1a\u6574\u4f53\u7ed3\u8bba\uff1b\u4f18\u5148\u5019\u9009\uff1b\u7406\u5316\u4e0e BLOSUM62 \u89e3\u8bfb\uff1b\u591a\u76ee\u6807/Pareto \u6743\u8861\uff1b\u5b9e\u9a8c\u9a8c\u8bc1\u987a\u5e8f\uff1b\u5c40\u9650\u6027\u3002
""".strip(),
    "en": r"""
You are the result-analysis module of FixedEnzymeAgent. Analyze only the supplied structured predictions and deterministic point-mutation descriptors. Never invent values.

Meaning of the inputs:
- EnzGFM activity_proxy is a relative/model proxy score, not experimental activity, kcat, Km or absolute enzyme activity.
- EpHod predicted optimum pH is computational.
- UniStab predicted ddG is computational; interpret direction only according to stability_direction.
- BLOSUM62 and hydrophobicity/volume/charge deltas only describe the amino-acid substitution and do not prove benefit or harm by themselves.
- pareto_optimal means the candidate is not simultaneously dominated by another passing candidate across normalized activity/pH/stability objectives.

Rules:
1. Do not describe predictions as experimental validation.
2. Do not modify, fabricate or infer unavailable numeric values.
3. Discuss complementary evidence and trade-offs across activity_proxy, pH, ddG, BLOSUM62, physicochemical deltas and Pareto status.
4. Explain that final_score is a ranking under the current user weights, not biological ground truth.
5. Recommend an experimental validation order and state that wet-lab validation is required.
6. Write plain English text; do not output a Markdown code fence.

Suggested structure: overall conclusion; priority candidates; physicochemical/BLOSUM62 interpretation; multi-objective/Pareto trade-offs; experimental order; limitations.
""".strip(),
}


def analyze_results(
    *,
    cfg: dict,
    ranked_csv: Path,
    wt_ph: float,
    client: DeepSeekClient,
) -> str:
    df = pd.read_csv(ranked_csv)
    # Never send raw WT/mutant protein sequences to the third-party LLM.
    allowed_columns = [
        "mutation",
        "position",
        "activity_proxy",
        "ph_opt",
        "delta_ph_from_wt",
        "ddg",
        "activity_component",
        "ph_component",
        "stability_component",
        "final_score",
        "passed",
        "reasons",
        "wt_aa",
        "mutant_aa",
        "wt_residue_class",
        "mutant_residue_class",
        "residue_class_change",
        "delta_hydrophobicity",
        "delta_volume",
        "delta_charge",
        "blosum62",
        "conservative_substitution",
        "pareto_optimal",
    ]
    existing = [c for c in allowed_columns if c in df.columns]
    limit = int(cfg.get("llm", {}).get("analysis_top_n", 20))
    rows = json.loads(df[existing].head(max(1, limit)).to_json(orient="records"))

    language = str(cfg.get("report", {}).get("language", "zh"))
    if language not in _SYSTEM_PROMPTS:
        language = "zh"

    payload = {
        "report_language": language,
        "user_request": cfg.get("user_request"),
        "selection": cfg["selection"],
        "candidate_pool_size": int(len(df)),
        "wt_predicted_optimum_ph": float(wt_ph),
        "top_candidates": rows,
    }
    llm_cfg = cfg["llm"]
    instruction = (
        "\u8bf7\u5206\u6790\u4e0b\u9762\u7684 FixedEnzymeAgent JSON \u7ed3\u679c\u3002\u6240\u6709\u6570\u503c\u53ea\u80fd\u5f15\u7528 JSON \u4e2d\u5df2\u6709\u503c\uff1a\n"
        if language == "zh"
        else "Analyze the following FixedEnzymeAgent JSON. You may only quote numeric values present in the JSON:\n"
    )
    return client.text(
        model=llm_cfg.get("analysis_model", "deepseek-v4-pro"),
        system_prompt=_SYSTEM_PROMPTS[language],
        user_prompt=instruction + json.dumps(payload, ensure_ascii=False, indent=2),
        max_tokens=int(llm_cfg.get("max_tokens", 4000)),
        thinking=bool(llm_cfg.get("thinking", True)),
        reasoning_effort=str(llm_cfg.get("reasoning_effort", "high")),
    )
