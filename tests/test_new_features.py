from __future__ import annotations

import csv
from pathlib import Path

from fixed_enzyme_agent.llm.parameter_parser import parse_user_request
from fixed_enzyme_agent.markdown_report import build_markdown_report
from fixed_enzyme_agent.mutation_analysis import annotate_candidates, mark_pareto_optimal
from fixed_enzyme_agent.types import Candidate
from fixed_enzyme_agent.visualization import generate_visualizations
from fixed_enzyme_agent.workflow import create_try_dir


class FakeDeepSeekClient:
    def json_object(self, **kwargs):
        return {
            "candidate": {
                "top_k": 5,
                "allowed_positions": None,
                "excluded_positions": None,
            },
            "selection": {
                "target_ph": 8.0,
                "ph_tolerance": 0.8,
                "min_activity_proxy": None,
                "stability_direction": None,
                "stability_operator": None,
                "stability_threshold": None,
                "weights": {"ph": 2, "activity": 5, "stability": 3},
                "top_k": 3,
            },
            "interpretation": "\u4f18\u5148\u6d3b\u6027\uff0c\u517c\u987e\u7a33\u5b9a\u6027\uff0c\u76ee\u6807 pH 8.0",
        }


def _base_cfg():
    return {
        "user_request": "\u76ee\u6807 pH 8.0\uff0c\u6d3b\u6027\u4f18\u5148\uff0c\u8fd4\u56de 3 \u4e2a",
        "candidate": {"top_k": 32, "allowed_positions": None, "excluded_positions": []},
        "selection": {
            "target_ph": 7.5,
            "ph_tolerance": 1.0,
            "min_activity_proxy": None,
            "stability_direction": "lower_is_better",
            "stability_operator": "none",
            "stability_threshold": None,
            "weights": {"ph": 0.3, "activity": 0.4, "stability": 0.3},
            "top_k": 10,
        },
        "llm": {"parameter_model": "deepseek-v4-flash", "max_tokens": 4000},
    }


def test_try_dir_sequence(tmp_path: Path):
    assert create_try_dir(tmp_path).name == "try1"
    assert create_try_dir(tmp_path).name == "try2"
    (tmp_path / "try5").mkdir()
    assert create_try_dir(tmp_path).name == "try6"


def test_llm_parameter_whitelist_merge():
    cfg, parsed = parse_user_request(_base_cfg(), client=FakeDeepSeekClient())
    assert parsed["interpretation"]
    assert cfg["candidate"]["top_k"] == 32
    assert cfg["selection"]["target_ph"] == 8.0
    assert cfg["selection"]["top_k"] == 3
    assert abs(sum(cfg["selection"]["weights"].values()) - 1.0) < 1e-9
    assert cfg["selection"]["stability_direction"] == "lower_is_better"


def test_physicochemical_blosum_and_pareto():
    candidates = [
        Candidate(
            mutation="A1V",
            wt_sequence="ACD",
            activity_component=0.9,
            ph_component=0.7,
            stability_component=0.8,
            passed=True,
        ),
        Candidate(
            mutation="C2S",
            wt_sequence="ACD",
            activity_component=0.8,
            ph_component=0.6,
            stability_component=0.7,
            passed=True,
        ),
        Candidate(
            mutation="D3E",
            wt_sequence="ACD",
            activity_component=0.7,
            ph_component=0.9,
            stability_component=0.9,
            passed=True,
        ),
    ]
    annotate_candidates(candidates)
    mark_pareto_optimal(candidates)

    first = candidates[0]
    assert first.position == 1
    assert first.wt_aa == "A"
    assert first.mutant_aa == "V"
    assert round(float(first.delta_hydrophobicity), 6) == 2.4
    assert round(float(first.delta_volume), 6) == 51.4
    assert first.blosum62 == 0
    assert first.conservative_substitution is True

    # C2S is dominated by A1V; A1V and D3E trade objectives and remain Pareto-optimal.
    assert candidates[0].pareto_optimal is True
    assert candidates[1].pareto_optimal is False
    assert candidates[2].pareto_optimal is True


def _write_result_csvs(tmp_path: Path) -> tuple[Path, Path]:
    fields = [
        "mutation", "position", "wt_aa", "mutant_aa", "activity_proxy", "ph_opt",
        "delta_ph_from_wt", "ddg", "activity_component", "ph_component",
        "stability_component", "final_score", "passed", "reasons", "blosum62",
        "conservative_substitution", "pareto_optimal", "wt_residue_class",
        "mutant_residue_class", "residue_class_change", "delta_hydrophobicity",
        "delta_volume", "delta_charge", "mutant_sequence",
    ]
    rows = [
        ["A1C", 1, "A", "C", 1.2, 7.5, 0.1, -0.4, 1.0, 1.0, 1.0, 1.0, True, "", 0, True, True,
         "nonpolar", "polar_uncharged", "nonpolar->polar_uncharged", 0.7, 19.9, 0.0, "CCD"],
        ["C2S", 2, "C", "S", 0.8, 7.8, 0.4, 0.1, 0.5, 0.7, 0.5, 0.56, True, "", -1, False, False,
         "polar_uncharged", "polar_uncharged", "polar_uncharged->polar_uncharged", -3.3, -19.5, 0.0, "ASD"],
    ]
    ranked = tmp_path / "02_candidate_pool_ranked.csv"
    selected = tmp_path / "03_selected_topk.csv"
    for path in [ranked, selected]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)
    return ranked, selected


def test_visualization_and_bilingual_markdown(tmp_path: Path):
    ranked, selected = _write_result_csvs(tmp_path)

    figures = generate_visualizations(
        ranked_csv=ranked,
        selected_csv=selected,
        output_dir=tmp_path / "figures",
        sequence_length=3,
    )
    assert len(figures) == 5
    assert all(p.is_file() and p.stat().st_size > 0 for p in figures)
    assert (tmp_path / "figures" / "04_mutation_sequence_map.png").is_file()
    assert (tmp_path / "figures" / "05_pareto_front.png").is_file()

    zh = build_markdown_report(
        run_dir=tmp_path,
        analysis_text="A1C \u7efc\u5408\u8868\u73b0\u6700\u4f73\u3002",
        figure_paths=figures,
        selected_csv=selected,
        user_request="\u76ee\u6807 pH 7.5",
        language="zh",
    )
    zh_text = zh.read_text(encoding="utf-8")
    assert zh.name == "report_zh.md"
    assert "\u5019\u9009\u7ed3\u679c\u5361" in zh_text
    assert "BLOSUM62" in zh_text
    assert "A1C \u7efc\u5408\u8868\u73b0\u6700\u4f73" in zh_text
    assert "figures/05_pareto_front.png" in zh_text

    en = build_markdown_report(
        run_dir=tmp_path,
        analysis_text="A1C is the strongest overall candidate.",
        figure_paths=figures,
        selected_csv=selected,
        user_request="Target pH 7.5",
        language="en",
    )
    en_text = en.read_text(encoding="utf-8")
    assert en.name == "report_en.md"
    assert "Candidate cards" in en_text
    assert "Pareto optimal" in en_text
    assert "A1C is the strongest overall candidate." in en_text
