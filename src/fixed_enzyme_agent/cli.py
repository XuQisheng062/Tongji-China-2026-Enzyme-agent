from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from .config import load_config


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _resolve_api_key(args, cfg: dict) -> str | None:
    llm_enabled = bool(cfg.get("llm", {}).get("enabled", False))
    if not llm_enabled:
        return None
    if getattr(args, "api_key", None):
        return args.api_key
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    if getattr(args, "ask_api_key", False):
        return getpass.getpass("DeepSeek API key: ")
    return None


def _apply_run_overrides(args, cfg: dict) -> None:
    request = getattr(args, "request", None)
    if request:
        cfg["user_request"] = request
        cfg.setdefault("llm", {})["enabled"] = True
    if getattr(args, "enable_llm", False):
        cfg.setdefault("llm", {})["enabled"] = True
    if getattr(args, "disable_llm", False):
        cfg.setdefault("llm", {})["enabled"] = False
    report_lang = getattr(args, "report_lang", None)
    if report_lang:
        cfg.setdefault("report", {})["language"] = report_lang


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request",
        help="\u81ea\u7136\u8bed\u8a00\u7b5b\u9009\u76ee\u6807\uff1b\u63d0\u4f9b\u540e\u81ea\u52a8\u542f\u7528 DeepSeek \u53c2\u6570\u89e3\u6790",
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key\uff08\u4f1a\u8fdb\u5165 shell history\uff1b\u66f4\u63a8\u8350\u73af\u5883\u53d8\u91cf\u6216 --ask-api-key\uff09",
    )
    parser.add_argument(
        "--ask-api-key",
        action="store_true",
        help="\u5728\u7ec8\u7aef\u5b89\u5168\u4ea4\u4e92\u8f93\u5165 DeepSeek API key\uff0c\u4e0d\u56de\u663e",
    )
    parser.add_argument("--enable-llm", action="store_true", help="\u5f3a\u5236\u542f\u7528 DeepSeek")
    parser.add_argument("--disable-llm", action="store_true", help="\u672c\u6b21\u8fd0\u884c\u5173\u95ed DeepSeek")
    parser.add_argument(
        "--print-llm",
        action="store_true",
        help="\u5c06 DeepSeek \u53c2\u6570\u89e3\u6790 JSON \u548c\u7ed3\u679c\u5206\u6790\u540c\u65f6\u8f93\u51fa\u5230\u7ec8\u7aef",
    )
    parser.add_argument(
        "--report-lang",
        choices=["zh", "en"],
        help="Markdown/DeepSeek \u5206\u6790\u8f93\u51fa\u8bed\u8a00\uff1b\u8986\u76d6 config.report.language",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Enzyme Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="\u8fd0\u884c\u5b8c\u6574\u56fa\u5b9a\u94fe\u8def")
    p_run.add_argument("config")
    p_run.add_argument("--cpu", action="store_true", help="\u5b8c\u6574\u94fe\u8def\u5f3a\u5236 CPU\uff08\u975e\u5e38\u6162\uff0c\u4ec5\u8c03\u8bd5\uff09")
    _add_run_args(p_run)

    p_cpu = sub.add_parser("validate-cpu", help="\u771f\u5b9e\u52a0\u8f7d EpHod/EnzGFM/UniStab \u5e76\u5728 CPU \u5404\u8dd1\u4e00\u6b21")
    p_cpu.add_argument("config")

    p_gpu = sub.add_parser("check-gpu", help="\u68c0\u67e5\u4e09\u4e2a\u6a21\u578b Python \u73af\u5883\u662f\u5426\u90fd\u80fd\u770b\u5230\u9009\u5b9a GPU")
    p_gpu.add_argument("config")

    p_paths = sub.add_parser("check-paths", help="\u53ea\u68c0\u67e5\u4ed3\u5e93\u3001\u5165\u53e3\u3001\u6a21\u578b\u6743\u91cd\u8def\u5f84")
    p_paths.add_argument("config")

    p_ds = sub.add_parser("deepseek-test", help="\u4ece\u547d\u4ee4\u884c\u6d4b\u8bd5 DeepSeek API\uff0c\u5e76\u628a\u54cd\u5e94\u6253\u5370\u5230\u7ec8\u7aef")
    p_ds.add_argument("--api-key")
    p_ds.add_argument("--ask-api-key", action="store_true")
    p_ds.add_argument("--model", default="deepseek-v4-flash", choices=["deepseek-v4-flash", "deepseek-v4-pro"])
    p_ds.add_argument("--prompt", default="\u8bf7\u53ea\u56de\u590d\uff1aDeepSeek API OK")
    p_ds.add_argument("--base-url", default="https://api.deepseek.com")

    p_tools = sub.add_parser("tools", help="List tools discovered from plug-in roots")
    p_tools.add_argument("--plugin-root", action="append", default=[])
    p_tools.add_argument("--config", help="Also register built-in tools from this configuration")

    p_plan = sub.add_parser("plan", help="Build and validate a capability-based workflow plan")
    p_plan.add_argument("--capability", action="append", required=True)
    p_plan.add_argument("--initial-artifact", action="append", default=["protein_sequence"])
    p_plan.add_argument("--plugin-root", action="append", default=[])
    p_plan.add_argument("--config", help="Also register built-in tools from this configuration")
    p_plan.add_argument("--request", default="")
    p_plan.add_argument("--deepseek", action="store_true")
    p_plan.add_argument("--model", default="deepseek-v4-flash")
    p_plan.add_argument("--api-key")
    p_plan.add_argument("--ask-api-key", action="store_true")
    p_plan.add_argument("--base-url", default="https://api.deepseek.com")

    args = parser.parse_args()

    if args.command in {"tools", "plan"}:
        from .planner import GoalSpec, WorkflowPlanner
        from .tools import ToolRegistry, register_configured_tools

        registry = ToolRegistry()
        if args.config:
            register_configured_tools(registry, load_config(args.config))
        plugin_roots = args.plugin_root or ["plugins"]
        registry.discover(Path(root) for root in plugin_roots)
        if args.command == "tools":
            _print_json(registry.catalog())
            return
        goal = GoalSpec(
            capabilities=tuple(args.capability),
            initial_artifacts=tuple(dict.fromkeys(args.initial_artifact)),
            interpretation=args.request,
        )
        planner = WorkflowPlanner(registry)
        if args.deepseek:
            from .llm.deepseek import DeepSeekClient

            key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not key and args.ask_api_key:
                key = getpass.getpass("DeepSeek API key: ")
            client = DeepSeekClient(api_key=key, base_url=args.base_url)
            plan = planner.with_llm(goal, client=client, model=args.model)
        else:
            plan = planner.deterministic(goal)
        _print_json(plan.to_dict())
        return

    if args.command == "deepseek-test":
        # Lazy import keeps API connectivity testing independent of pandas/matplotlib/workflow imports.
        from .llm.deepseek import DeepSeekClient

        key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key and args.ask_api_key:
            key = getpass.getpass("DeepSeek API key: ")
        client = DeepSeekClient(api_key=key, base_url=args.base_url)
        text = client.text(
            model=args.model,
            system_prompt="You are a concise API connectivity tester.",
            user_prompt=args.prompt,
            max_tokens=256,
            thinking=False,
        )
        print(text)
        return

    cfg = load_config(args.config)

    if args.command == "run":
        from .workflow import FixedEnzymeWorkflow

        _apply_run_overrides(args, cfg)
        api_key = _resolve_api_key(args, cfg)
        report = FixedEnzymeWorkflow(
            cfg,
            force_cpu=args.cpu,
            deepseek_api_key=api_key,
            print_llm=args.print_llm,
        ).run()
        _print_json(report)
    elif args.command == "validate-cpu":
        from .validation import validate_three_models_cpu

        _print_json(validate_three_models_cpu(cfg))
    elif args.command == "check-gpu":
        from .preflight import gpu_preflight

        result = gpu_preflight(cfg)
        _print_json(result)
        if not all(v.get("ok", False) for v in result.values()):
            raise SystemExit(2)
    elif args.command == "check-paths":
        from .preflight import check_paths

        errors = check_paths(cfg)
        _print_json({"ok": not errors, "errors": errors})
        if errors:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
