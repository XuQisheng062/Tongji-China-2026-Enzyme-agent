from __future__ import annotations

import copy
import csv
import json
import shutil
import traceback
from pathlib import Path

from .adapters import EpHodAdapter, EnzGFMAdapter, UniStabAdapter
from .config import validate_candidate_and_selection
from .codon_optimization import resolve_codon_optimization_request, write_codon_optimization_outputs
from .io_utils import write_candidates_csv, write_json
from .markdown_report import build_markdown_report
from .mutation_analysis import annotate_candidates, mark_pareto_optimal, write_mutation_features_csv
from .mutations import apply_mutation, generate_single_substitutions
from .runner import build_env
from .scoring import rank_candidates
from .types import Candidate
from .visualization import generate_visualizations


STAGES = [
    "resolve_user_request",
    "generate_mutations",
    "enzgfm_scan_and_candidate_selection",
    "ephod_ph_prediction",
    "unistab_stability_prediction",
    "personalized_filter_and_rank",
    "point_mutation_feature_analysis",
    "codon_optimization",
    "visualize_results",
    "deepseek_result_analysis",
    "write_markdown_report",
]


def create_try_dir(out_dir: Path) -> Path:
    """Atomically create try1, try2, ... under an output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in out_dir.iterdir():
        if path.is_dir() and path.name.startswith("try"):
            suffix = path.name[3:]
            if suffix.isdigit():
                numbers.append(int(suffix))
    next_number = max(numbers, default=0) + 1
    while True:
        run_dir = out_dir / f"try{next_number}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            next_number += 1


class FixedEnzymeWorkflow:
    def __init__(
        self,
        cfg: dict,
        *,
        force_cpu: bool = False,
        deepseek_api_key: str | None = None,
        print_llm: bool = False,
    ):
        self.cfg = cfg
        self.force_cpu = force_cpu
        self.deepseek_api_key = deepseek_api_key
        self.print_llm = print_llm
        m = cfg["models"]
        self.ephod = EpHodAdapter(m["ephod"]["repo"], m["ephod"]["python"])
        self.enzgfm = EnzGFMAdapter(
            m["enzgfm"]["repo"],
            m["enzgfm"]["python"],
            m["enzgfm"]["model_location"],
        )
        self.unistab = UniStabAdapter(
            m["unistab"]["repo"],
            m["unistab"]["python"],
            m["unistab"]["checkpoint"],
        )

    def _path_preflight(self) -> None:
        errors: list[str] = []
        for adapter in [self.ephod, self.enzgfm, self.unistab]:
            errors.extend(adapter.preflight())
        if errors:
            raise RuntimeError("\u6a21\u578b/\u8def\u5f84\u9884\u68c0\u5931\u8d25:\n- " + "\n- ".join(errors))

    def _deepseek_client(self, cfg: dict):
        from .llm.deepseek import DeepSeekClient

        return DeepSeekClient(
            api_key=self.deepseek_api_key,
            base_url=cfg["llm"].get("base_url", "https://api.deepseek.com"),
        )

    def run(self) -> dict:
        cfg = copy.deepcopy(self.cfg)
        base_out_dir = Path(cfg["output_dir"])
        run_dir = create_try_dir(base_out_dir)
        report_language = str(cfg.get("report", {}).get("language", "zh"))
        report: dict = {
            "stages": [],
            "force_cpu": self.force_cpu,
            "base_output_dir": str(base_out_dir),
            "run_dir": str(run_dir),
            "report_language": report_language,
            "status": "running",
        }
        success = False

        try:
            parsed_request: dict = {}
            llm_cfg = cfg.get("llm", {})
            if bool(llm_cfg.get("enabled", False)) and bool(llm_cfg.get("parameter_parsing", True)):
                from .llm.parameter_parser import parse_user_request

                client = self._deepseek_client(cfg)
                cfg, parsed_request = parse_user_request(cfg, client=client)
                if self.print_llm:
                    print("\n[DeepSeek \u53c2\u6570\u89e3\u6790]")
                    print(json.dumps(parsed_request, ensure_ascii=False, indent=2))
                report["stages"].append({
                    "stage": STAGES[0],
                    "enabled": True,
                    "interpretation": parsed_request.get("interpretation"),
                })
            else:
                validate_candidate_and_selection(cfg, strict_ph_target=True)
                report["stages"].append({"stage": STAGES[0], "enabled": False})

            # Codon optimization is triggered deterministically from the original user request.
            # It is intentionally independent of the LLM so DeepSeek cannot invent a host species.
            codon_request = resolve_codon_optimization_request(cfg.get("user_request"))

            write_json(
                run_dir / "00_resolved_request.json",
                {
                    "user_request": cfg.get("user_request"),
                    "llm_enabled": bool(llm_cfg.get("enabled", False)),
                    "llm_parsed": parsed_request,
                    "resolved_candidate": cfg["candidate"],
                    "resolved_selection": cfg["selection"],
                    "codon_optimization": codon_request,
                    "report_language": report_language,
                },
            )

            self._path_preflight()
            seq = cfg["sequence"]
            runtime = cfg["runtime"]
            env = build_env(
                force_cpu=self.force_cpu,
                gpu_id=int(runtime["gpu_id"]),
                cache_dir=runtime["cache_dir"],
            )
            timeout = int(runtime["timeout_seconds"])

            # 1) Deterministic enumeration; no model yet.
            mutations = generate_single_substitutions(
                seq,
                allowed_positions=cfg["candidate"].get("allowed_positions"),
                excluded_positions=cfg["candidate"].get("excluded_positions"),
            )
            report["stages"].append({"stage": STAGES[1], "count": len(mutations)})

            # 2) EnzGFM is both candidate finder and mutation-effect proxy.
            activity = self.enzgfm.score_mutations(
                seq,
                mutations,
                work_dir=run_dir / "enzgfm_scan",
                env=env,
                timeout=timeout,
                force_cpu=self.force_cpu,
            )
            scan_sorted = sorted(mutations, key=lambda m: activity[m], reverse=True)
            scan_csv = run_dir / "01_enzgfm_full_scan.csv"
            with scan_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["mutation", "activity_proxy"])
                for mutation in scan_sorted:
                    writer.writerow([mutation, activity[mutation]])

            pool_n = min(int(cfg["candidate"]["top_k"]), len(scan_sorted))
            pool_mutations = scan_sorted[:pool_n]
            candidates = [
                Candidate(mutation=m, wt_sequence=seq, activity_proxy=activity[m])
                for m in pool_mutations
            ]
            for candidate in candidates:
                candidate.mutant_sequence = apply_mutation(seq, candidate.mutation)
            # Static amino-acid descriptors can be computed before the downstream models.
            annotate_candidates(candidates)
            report["stages"].append({
                "stage": STAGES[2],
                "pool_size": pool_n,
                "full_scan": len(scan_sorted),
            })

            # 3) EpHod: WT + pool mutants in one FASTA/model load.
            ph_input = {"WT": seq}
            ph_input.update({
                c.mutation: c.mutant_sequence
                for c in candidates
                if c.mutant_sequence
            })
            ph = self.ephod.predict(
                ph_input,
                work_dir=run_dir / "ephod",
                env=env,
                timeout=timeout,
            )
            wt_ph = ph["WT"]
            for candidate in candidates:
                candidate.ph_opt = ph[candidate.mutation]
                candidate.delta_ph_from_wt = candidate.ph_opt - wt_ph
            report["stages"].append({
                "stage": STAGES[3],
                "wt_ph": wt_ph,
                "count": len(candidates),
            })

            # 4) UniStab: batch input in original candidate order.
            ddg = self.unistab.predict_ddg(
                seq,
                {
                    c.mutation: c.mutant_sequence
                    for c in candidates
                    if c.mutant_sequence
                },
                work_dir=run_dir / "unistab",
                env=env,
                timeout=timeout,
                force_cpu=self.force_cpu,
            )
            for candidate in candidates:
                candidate.ddg = ddg[candidate.mutation]
            report["stages"].append({"stage": STAGES[4], "count": len(candidates)})

            # 5) Existing deterministic personalized ranking.
            ranked = rank_candidates(candidates, cfg["selection"])
            report["stages"].append({
                "stage": STAGES[5],
                "passed": sum(1 for c in ranked if c.passed),
            })

            # 6) Add 3-objective Pareto status after normalized scoring components exist.
            mark_pareto_optimal(ranked)
            passed = [c for c in ranked if c.passed]
            selected = passed[: int(cfg["selection"]["top_k"])]
            report["stages"].append({
                "stage": STAGES[6],
                "feature_count": len(ranked),
                "pareto_optimal": sum(1 for c in ranked if c.pareto_optimal),
                "selected": len(selected),
            })

            ranked_csv = run_dir / "02_candidate_pool_ranked.csv"
            selected_csv = run_dir / "03_selected_topk.csv"
            features_csv = run_dir / "04_mutation_features.csv"
            write_candidates_csv(ranked_csv, ranked)
            write_candidates_csv(selected_csv, selected)
            write_mutation_features_csv(features_csv, ranked)

            # 7) Optional codon optimization is a deterministic post-processing step.
            # Built-in hosts are offline; non-built-in hosts can query/cache Kazusa.
            # It runs only when the user explicitly mentions codon optimization.
            codon_info = write_codon_optimization_outputs(
                run_dir=run_dir,
                sequence_name=str(cfg.get("sequence_name", "enzyme")),
                wt_sequence=seq,
                selected=selected,
                resolved_request=codon_request,
                cache_dir=cfg.get("runtime", {}).get("cache_dir"),
            )
            report["stages"].append({
                "stage": STAGES[7],
                "enabled": bool(codon_info.get("enabled")),
                "status": codon_info.get("status"),
                "host_count": codon_info.get("host_count", 0),
                "organisms": [
                    host.get("scientific_name")
                    for host in codon_info.get("resolved_hosts", [])
                ],
                "count": codon_info.get("count", 0),
                "failed_hosts": codon_info.get("failed_hosts", []),
                "warnings": codon_info.get("warnings", []),
            })

            # 8) Existing plots + mutation sequence map + Pareto plot.
            figure_paths: list[Path] = []
            if bool(cfg.get("visualization", {}).get("enabled", True)):
                figure_paths = generate_visualizations(
                    ranked_csv=ranked_csv,
                    selected_csv=selected_csv,
                    output_dir=run_dir / "figures",
                    sequence_length=len(seq),
                )
            report["stages"].append({
                "stage": STAGES[8],
                "enabled": bool(cfg.get("visualization", {}).get("enabled", True)),
                "figures": [str(p) for p in figure_paths],
            })

            # 8) DeepSeek now also receives deterministic mutation descriptors/Pareto state.
            analysis_text = (
                "DeepSeek \u7ed3\u679c\u5206\u6790\u672a\u542f\u7528\u3002"
                if report_language == "zh"
                else "DeepSeek result analysis is disabled."
            )
            analysis_error: str | None = None
            if bool(llm_cfg.get("enabled", False)) and bool(llm_cfg.get("result_analysis", True)):
                from .llm.result_analyzer import analyze_results

                try:
                    client = self._deepseek_client(cfg)
                    analysis_text = analyze_results(
                        cfg=cfg,
                        ranked_csv=ranked_csv,
                        wt_ph=wt_ph,
                        client=client,
                    )
                    if self.print_llm:
                        print("\n[DeepSeek \u7ed3\u679c\u5206\u6790]")
                        print(analysis_text)
                except Exception as exc:
                    analysis_error = f"{type(exc).__name__}: {exc}"
                    if not bool(llm_cfg.get("analysis_fail_open", True)):
                        raise
                    if report_language == "zh":
                        analysis_text = (
                            "DeepSeek \u7ed3\u679c\u5206\u6790\u8c03\u7528\u5931\u8d25\uff0c\u4f46\u4e09\u4e2a\u9884\u6d4b\u6a21\u578b\u3001\u6392\u5e8f\u3001\u70b9\u7a81\u53d8\u5206\u6790\u4e0e\u53ef\u89c6\u5316\u7ed3\u679c\u5df2\u7ecf\u4fdd\u7559\u3002\n"
                            f"\u9519\u8bef: {analysis_error}"
                        )
                    else:
                        analysis_text = (
                            "DeepSeek result analysis failed, but the three-model predictions, ranking, "
                            "point-mutation analysis and visualizations were retained.\n"
                            f"Error: {analysis_error}"
                        )
            (run_dir / "analysis.txt").write_text(analysis_text, encoding="utf-8")
            report["stages"].append({
                "stage": STAGES[9],
                "enabled": bool(llm_cfg.get("enabled", False) and llm_cfg.get("result_analysis", True)),
                "error": analysis_error,
            })

            # 9) Emit exactly one chosen Markdown language per run.
            report_md = build_markdown_report(
                run_dir=run_dir,
                analysis_text=analysis_text,
                figure_paths=figure_paths,
                selected_csv=selected_csv,
                user_request=cfg.get("user_request"),
                language=report_language,
                codon_optimization=codon_info,
            )
            report["stages"].append({"stage": STAGES[10], "file": str(report_md)})

            report["selected"] = [c.to_dict() for c in selected]
            report["candidate_pool_size"] = len(candidates)
            report["wt_ph"] = wt_ph
            report["mutation_features_file"] = str(features_csv)
            report["codon_optimization"] = codon_info
            report["analysis_file"] = str(run_dir / "analysis.txt")
            report["markdown_report"] = str(report_md)
            report["figures"] = [str(p) for p in figure_paths]
            report["status"] = "success"
            write_json(run_dir / "result.json", report)
            success = True
            return report
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            write_json(run_dir / "result.json", report)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            # tryN is always retained. keep_temp only controls large raw model intermediates
            # after a successful run; failed runs keep everything for debugging.
            if success and not bool(cfg.get("runtime", {}).get("keep_temp", True)):
                for name in ["enzgfm_scan", "ephod", "unistab"]:
                    shutil.rmtree(run_dir / name, ignore_errors=True)
            print(f"\u672c\u6b21\u8fd0\u884c\u76ee\u5f55: {run_dir}")
