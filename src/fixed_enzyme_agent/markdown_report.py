from __future__ import annotations

from pathlib import Path

import pandas as pd


_TEXT = {
    "zh": {
        "title": "FixedEnzymeAgent \u5206\u6790\u62a5\u544a",
        "goal": "\u7528\u6237\u76ee\u6807",
        "no_goal": "\u672a\u63d0\u4f9b\u81ea\u7136\u8bed\u8a00\u76ee\u6807\uff1b\u4f7f\u7528\u914d\u7f6e\u6587\u4ef6\u4e2d\u7684\u7b5b\u9009\u53c2\u6570\u3002",
        "selected": "\u6700\u7ec8\u5019\u9009",
        "no_table": "\u65e0\u6cd5\u8bfb\u53d6\u6700\u7ec8\u5019\u9009\u8868\u3002",
        "no_candidates": "\u6ca1\u6709\u5019\u9009\u901a\u8fc7\u5f53\u524d\u7b5b\u9009\u6761\u4ef6\u3002",
        "visuals": "\u53ef\u89c6\u5316\u7ed3\u679c",
        "cards": "\u5019\u9009\u7ed3\u679c\u5361",
        "deepseek": "DeepSeek \u7ed3\u679c\u5206\u6790",
        "no_analysis": "\u672a\u751f\u6210 DeepSeek \u5206\u6790\u3002",
        "files": "\u7ed3\u679c\u6587\u4ef6",
        "limits": "\u89e3\u91ca\u8fb9\u754c",
        "rank": "\u6392\u540d",
        "pareto": "Pareto \u6700\u4f18",
        "yes": "\u662f",
        "no": "\u5426",
        "mutation": "\u7a81\u53d8",
        "position": "\u4f4d\u7f6e",
        "final_score": "\u7efc\u5408\u5206\u6570",
        "activity": "EnzGFM activity_proxy",
        "ph": "EpHod \u6700\u9002 pH",
        "ddg": "UniStab ddG",
        "blosum": "BLOSUM62",
        "hydro": "\u758f\u6c34\u6027\u53d8\u5316 \u0394",
        "volume": "\u6b8b\u57fa\u4f53\u79ef\u53d8\u5316 \u0394 (\u00c5\u00b3)",
        "charge": "\u540d\u4e49\u4fa7\u94fe\u7535\u8377\u53d8\u5316 \u0394",
        "class": "\u6b8b\u57fa\u7c7b\u522b\u53d8\u5316",
        "conservative": "BLOSUM62 \u975e\u8d1f\u66ff\u6362",
        "reason": "\u8fc7\u6ee4\u539f\u56e0",
        "none": "\u65e0",
        "codon": "\u5bc6\u7801\u5b50\u4f18\u5316",
        "codon_not_requested": "\u672c\u6b21\u672a\u8bf7\u6c42\u5bc6\u7801\u5b50\u4f18\u5316\u3002",
    },
    "en": {
        "title": "FixedEnzymeAgent Analysis Report",
        "goal": "User goal",
        "no_goal": "No natural-language goal was provided; selection parameters came from the config file.",
        "selected": "Selected candidates",
        "no_table": "Unable to read the selected-candidate table.",
        "no_candidates": "No candidate passed the current filters.",
        "visuals": "Visualizations",
        "cards": "Candidate cards",
        "deepseek": "DeepSeek result analysis",
        "no_analysis": "No DeepSeek analysis was generated.",
        "files": "Output files",
        "limits": "Interpretation boundaries",
        "rank": "Rank",
        "pareto": "Pareto optimal",
        "yes": "Yes",
        "no": "No",
        "mutation": "Mutation",
        "position": "Position",
        "final_score": "Final score",
        "activity": "EnzGFM activity_proxy",
        "ph": "EpHod optimum pH",
        "ddg": "UniStab ddG",
        "blosum": "BLOSUM62",
        "hydro": "Hydrophobicity \u0394",
        "volume": "Residue volume \u0394 (\u00c5\u00b3)",
        "charge": "Nominal side-chain charge \u0394",
        "class": "Residue-class change",
        "conservative": "Non-negative BLOSUM62 substitution",
        "reason": "Filter reason",
        "none": "None",
        "codon": "Codon optimization",
        "codon_not_requested": "Codon optimization was not requested for this run.",
    },
}

_FIGURE_CAPTIONS = {
    "zh": {
        "01_final_score": "\u6700\u7ec8\u5019\u9009\u7efc\u5408\u8bc4\u5206",
        "02_activity_vs_stability": "\u6d3b\u6027\u4ee3\u7406\u4e0e\u7a33\u5b9a\u6027\u9884\u6d4b\u5173\u7cfb",
        "03_property_heatmap": "\u5019\u9009\u591a\u6307\u6807\u70ed\u56fe",
        "04_mutation_sequence_map": "\u7a81\u53d8\u4f4d\u70b9\u5e8f\u5217\u5206\u5e03\u56fe",
        "05_pareto_front": "\u4e09\u76ee\u6807 Pareto \u524d\u6cbf",
    },
    "en": {
        "01_final_score": "Final scores of selected candidates",
        "02_activity_vs_stability": "Activity proxy versus stability prediction",
        "03_property_heatmap": "Multi-property heatmap",
        "04_mutation_sequence_map": "Mutation sequence map",
        "05_pareto_front": "Three-objective Pareto front",
    },
}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _bool_text(value, language: str) -> str:
    text = _TEXT[language]
    if isinstance(value, str):
        flag = value.strip().lower() in {"true", "1", "yes"}
    else:
        flag = bool(value)
    return text["yes"] if flag else text["no"]


def _selected_table(selected_csv: Path, language: str) -> str:
    text = _TEXT[language]
    try:
        df = pd.read_csv(selected_csv)
    except Exception:
        return text["no_table"]
    if df.empty:
        return text["no_candidates"]

    columns = [
        ("mutation", text["mutation"]),
        ("activity_proxy", "activity_proxy"),
        ("ph_opt", "pH opt"),
        ("ddg", "ddG"),
        ("blosum62", "BLOSUM62"),
        ("pareto_optimal", text["pareto"]),
        ("final_score", text["final_score"]),
    ]
    columns = [(key, label) for key, label in columns if key in df.columns]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        values: list[str] = []
        for key, _ in columns:
            if key == "pareto_optimal":
                values.append(_bool_text(row[key], language))
            elif key in {"activity_proxy", "ph_opt", "ddg", "final_score"}:
                values.append(_fmt(row[key]))
            else:
                values.append(str(row[key]))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def _candidate_cards(selected_csv: Path, language: str) -> str:
    text = _TEXT[language]
    try:
        df = pd.read_csv(selected_csv)
    except Exception:
        return text["no_table"]
    if df.empty:
        return text["no_candidates"]

    blocks: list[str] = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        mutation = str(row.get("mutation", f"candidate-{rank}"))
        pareto = _bool_text(row.get("pareto_optimal", False), language)
        blocks.extend([
            f"### #{rank} {mutation}",
            "",
            f"**{text['pareto']}**: {pareto}",
            "",
            f"| {text['mutation']} metric | Value |",
            "| --- | ---: |",
            f"| {text['position']} | {row.get('position', '')} |",
            f"| {text['final_score']} | {_fmt(row.get('final_score'))} |",
            f"| {text['activity']} | {_fmt(row.get('activity_proxy'))} |",
            f"| {text['ph']} | {_fmt(row.get('ph_opt'))} |",
            f"| {text['ddg']} | {_fmt(row.get('ddg'))} |",
            f"| {text['blosum']} | {row.get('blosum62', '')} |",
            f"| {text['hydro']} | {_fmt(row.get('delta_hydrophobicity'))} |",
            f"| {text['volume']} | {_fmt(row.get('delta_volume'))} |",
            f"| {text['charge']} | {_fmt(row.get('delta_charge'))} |",
            f"| {text['class']} | {row.get('residue_class_change', '')} |",
            f"| {text['conservative']} | {_bool_text(row.get('conservative_substitution', False), language)} |",
            f"| {text['reason']} | {row.get('reasons', '') or text['none']} |",
            "",
        ])
    return "\n".join(blocks)


def build_markdown_report(
    *,
    run_dir: Path,
    analysis_text: str,
    figure_paths: list[Path],
    selected_csv: Path,
    user_request: str | None,
    language: str = "zh",
    codon_optimization: dict | None = None,
) -> Path:
    if language not in _TEXT:
        raise ValueError("report language must be 'zh' or 'en'")
    text = _TEXT[language]
    captions = _FIGURE_CAPTIONS[language]

    lines = [
        f"# {text['title']}",
        "",
        f"## {text['goal']}",
        "",
        user_request.strip() if user_request and user_request.strip() else text["no_goal"],
        "",
        f"## {text['selected']}",
        "",
        _selected_table(selected_csv, language),
        "",
        f"## {text['cards']}",
        "",
        _candidate_cards(selected_csv, language),
        "",
        f"## {text['visuals']}",
        "",
    ]

    for image in figure_paths:
        relative = image.relative_to(run_dir).as_posix()
        caption = captions.get(image.stem, image.stem)
        lines.extend([
            f"### {caption}",
            "",
            f"![{caption}]({relative})",
            "",
        ])

    lines.extend([f"## {text['codon']}", ""])
    codon = codon_optimization or {}
    if codon.get("enabled"):
        hosts = codon.get("resolved_hosts", []) or []
        failed_hosts = codon.get("failed_hosts", []) or []
        if language == "zh":
            lines.extend([
                f"- \u72b6\u6001\uff1a`{codon.get('status', 'unknown')}`",
                f"- \u6210\u529f\u5bbf\u4e3b\u6570\uff1a**{len(hosts)}**\uff1b\u6bcf\u4e2a\u5bbf\u4e3b\u8f93\u51fa **{codon.get('sequences_per_host', 0)}** \u6761\u86cb\u767d\u5bf9\u5e94 CDS\u3002",
                "- \u65b9\u6cd5\uff1a\u6309\u76ee\u6807\u5bbf\u4e3b\u5bc6\u7801\u5b50\u4f7f\u7528\u8868\uff0c\u4e3a\u6bcf\u4e2a\u6c28\u57fa\u9178\u786e\u5b9a\u6027\u9009\u62e9\u6700\u9ad8\u9891\u540c\u4e49\u5bc6\u7801\u5b50\uff1b\u5c5e\u4e8e\u57fa\u7840\u5bc6\u7801\u5b50\u504f\u597d\u4f18\u5316\u3002",
                "- 11 \u4e2a\u5e38\u7528\u5bbf\u4e3b\u4f7f\u7528\u5185\u7f6e Kazusa \u6765\u6e90\u8868\uff1b\u5176\u4ed6\u62c9\u4e01\u5b66\u540d/TaxID \u901a\u8fc7 Kazusa \u52a8\u6001\u67e5\u8be2\u5e76\u7f13\u5b58\u3002",
                "- FASTA \u8f93\u51fa\u5305\u542b\u7ec8\u6b62\u5bc6\u7801\u5b50\uff1bCSV \u540c\u65f6\u63d0\u4f9b\u4e0d\u542b\u548c\u5305\u542b\u7ec8\u6b62\u5bc6\u7801\u5b50\u7684 CDS\u3002",
            ])
            if hosts:
                lines.extend(["", "\u76ee\u6807\u5bbf\u4e3b\uff1a"])
                for host in hosts:
                    label = host.get("display_name_zh") or host.get("scientific_name")
                    lines.append(
                        f"- **{label}** (`{host.get('scientific_name')}`, NCBI TaxID {host.get('taxid')})\uff1b\u6765\u6e90 `{host.get('codon_source')}`"
                    )
            for warning in codon.get("warnings", []) or []:
                lines.append(f"- \u6ce8\u610f\uff1a{warning}")
            for failed in failed_hosts:
                lines.append(f"- \u672a\u5b8c\u6210\u5bbf\u4e3b `{failed.get('query')}`\uff1a{failed.get('error')}")
        else:
            lines.extend([
                f"- Status: `{codon.get('status', 'unknown')}`",
                f"- Successfully resolved hosts: **{len(hosts)}**; **{codon.get('sequences_per_host', 0)}** protein-derived CDS records per host.",
                "- Method: deterministic back-translation using the most frequent synonymous codon for each amino acid in the target host codon-usage table.",
                "- Eleven common hosts use built-in Kazusa-derived tables; other Latin scientific names/TaxIDs are queried dynamically from Kazusa and cached.",
                "- FASTA outputs include a stop codon; the CSV contains CDS versions both with and without the stop codon.",
            ])
            if hosts:
                lines.extend(["", "Expression hosts:"])
                for host in hosts:
                    lines.append(
                        f"- **{host.get('scientific_name')}** (NCBI TaxID {host.get('taxid')}); source `{host.get('codon_source')}`"
                    )
            for warning in codon.get("warnings", []) or []:
                lines.append(f"- Note: {warning}")
            for failed in failed_hosts:
                lines.append(f"- Unresolved host `{failed.get('query')}`: {failed.get('error')}")
        lines.append("")
    else:
        lines.extend([text["codon_not_requested"], ""])

    lines.extend([
        f"## {text['deepseek']}",
        "",
        analysis_text.strip() if analysis_text.strip() else text["no_analysis"],
        "",
        f"## {text['files']}",
        "",
    ])

    if language == "zh":
        lines.extend([
            "- `00_resolved_request.json`\uff1a\u81ea\u7136\u8bed\u8a00\u9700\u6c42\u4e0e\u6700\u7ec8\u751f\u6548\u53c2\u6570",
            "- `01_enzgfm_full_scan.csv`\uff1aEnzGFM \u5168\u626b\u63cf\u7ed3\u679c",
            "- `02_candidate_pool_ranked.csv`\uff1a\u5019\u9009\u6c60\u7efc\u5408\u6392\u5e8f\u4e0e\u70b9\u7a81\u53d8\u5c5e\u6027",
            "- `03_selected_topk.csv`\uff1a\u6700\u7ec8 Top-K",
            "- `04_mutation_features.csv`\uff1a\u7406\u5316\u6027\u8d28\u3001BLOSUM62 \u4e0e Pareto \u6807\u8bb0",
            "- `05_codon_optimized_sequences.csv/.fasta`\uff1a\u591a\u5bbf\u4e3b\u5408\u5e76\u8f93\u51fa\uff1b\u4ec5\u5728\u660e\u786e\u8bf7\u6c42\u5bc6\u7801\u5b50\u4f18\u5316\u65f6\u751f\u6210",
            "- `analysis.txt`\uff1aDeepSeek \u7eaf\u6587\u672c\u5206\u6790",
            "- `figures/`\uff1a\u7edf\u8ba1\u56fe\u3001\u5e8f\u5217\u7a81\u53d8\u56fe\u4e0e Pareto \u56fe",
            "- `result.json`\uff1a\u7ed3\u6784\u5316\u8fd0\u884c\u62a5\u544a",
            "",
            f"## {text['limits']}",
            "",
            "- `activity_proxy` \u662f EnzGFM \u6a21\u578b\u76f8\u5bf9\u8bc4\u5206\uff0c\u4e0d\u662f\u5b9e\u9a8c\u6d3b\u6027\u3001kcat \u6216 Km\u3002",
            "- EpHod pH \u4e0e UniStab ddG \u5747\u4e3a\u8ba1\u7b97\u9884\u6d4b\u3002",
            "- BLOSUM62 \u4e0e\u7406\u5316\u6027\u8d28\u53d8\u5316\u53ea\u7528\u4e8e\u63cf\u8ff0\u66ff\u6362\u672c\u8eab\uff0c\u4e0d\u80fd\u5355\u72ec\u8bc1\u660e\u7a81\u53d8\u6709\u76ca\u3002",
            "- Pareto \u6700\u4f18\u8868\u793a\u5019\u9009\u5728 activity/pH/stability \u4e09\u4e2a\u5f52\u4e00\u5316\u76ee\u6807\u4e0a\u672a\u88ab\u5176\u4ed6\u5408\u683c\u5019\u9009\u540c\u65f6\u652f\u914d\u3002",
            "- `final_score` \u53d6\u51b3\u4e8e\u5f53\u524d\u7528\u6237\u6743\u91cd\u4e0e\u8fc7\u6ee4\u89c4\u5219\uff0c\u4e0d\u4ee3\u8868\u751f\u7269\u5b66\u771f\u503c\u3002",
            "- \u5f53\u524d\u5bc6\u7801\u5b50\u4f18\u5316\u4ec5\u57fa\u4e8e\u5bbf\u4e3b\u9ad8\u9891\u540c\u4e49\u5bc6\u7801\u5b50\uff1b\u52a8\u6001 Kazusa \u515c\u5e95\u4ec5\u652f\u6301\u6807\u51c6\u9057\u4f20\u5bc6\u7801\u8868 1\uff0c\u4e14\u4e0d\u5904\u7406\u9650\u5236\u6027\u5185\u5207\u9176\u4f4d\u70b9\u3001\u91cd\u590d\u5e8f\u5217\u3001mRNA \u4e8c\u7ea7\u7ed3\u6784\u7b49\u5408\u6210\u7ea6\u675f\u3002",
            "- \u6700\u7ec8\u5019\u9009\u5fc5\u987b\u7ecf\u8fc7\u5b9e\u9a8c\u9a8c\u8bc1\u3002",
            "",
        ])
    else:
        lines.extend([
            "- `00_resolved_request.json`: natural-language request and resolved parameters",
            "- `01_enzgfm_full_scan.csv`: full EnzGFM mutation scan",
            "- `02_candidate_pool_ranked.csv`: ranked pool with point-mutation descriptors",
            "- `03_selected_topk.csv`: final Top-K",
            "- `04_mutation_features.csv`: physicochemical changes, BLOSUM62 and Pareto flags",
            "- `05_codon_optimized_sequences.csv/.fasta`: combined multi-host output generated only when codon optimization is explicitly requested",
            "- `analysis.txt`: plain-text DeepSeek analysis",
            "- `figures/`: score plots, mutation sequence map and Pareto visualization",
            "- `result.json`: structured run report",
            "",
            f"## {text['limits']}",
            "",
            "- `activity_proxy` is an EnzGFM relative model score, not experimental activity, kcat or Km.",
            "- EpHod optimum pH and UniStab ddG are computational predictions.",
            "- BLOSUM62 and physicochemical deltas describe the substitution and do not by themselves prove benefit.",
            "- Pareto optimal means the candidate is not simultaneously dominated by another passing candidate across normalized activity/pH/stability objectives.",
            "- `final_score` depends on the active filters and user weights and is not biological ground truth.",
            "- Current codon optimization uses host-preferred synonymous codons only; dynamic Kazusa fallback assumes Standard genetic code 1 and does not optimize restriction sites, repeats, mRNA secondary structure, or other synthesis constraints.",
            "- Final candidates require experimental validation.",
            "",
        ])

    filename = "report_zh.md" if language == "zh" else "report_en.md"
    path = run_dir / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
