from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .schemas import VerificationResult, VerifiedExecutionError


def serializable(value):
    if hasattr(value, "to_dict"):
        return serializable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()
                if not any(secret in str(k).lower() for secret in ("api_key", "password", "secret", "token"))}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class VerificationGate:
    def __init__(self, config, run_dir, *, reflector=None, task=None):
        self.config = copy.deepcopy(config)
        self.task = copy.deepcopy(task or {})
        self.reflector = reflector
        self.max_retries = int(config.get("reflection", {}).get("max_retries", 2))
        if self.max_retries < 0:
            raise ValueError("reflection.max_retries must be non-negative")
        self.used_retries = 0
        self.history = []
        self.task_id = uuid.uuid4().hex
        self.path = Path(run_dir) / "trajectories" / f"{self.task_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event):
        self.history.append(copy.deepcopy(event))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable(event), ensure_ascii=False, allow_nan=False) + "\n")

    def run(self, action, attempt, repairs):
        current = copy.deepcopy(action)
        seen = set()
        retry_index = 0
        while True:
            fingerprint = json.dumps(current, sort_keys=True, default=str)
            seen.add(fingerprint)
            started = time.perf_counter()
            value, verdict, status = attempt(copy.deepcopy(current))
            options = repairs(current, verdict) if not verdict.passed else []
            options = [option for option in options if
                       json.dumps(option["action"], sort_keys=True, default=str) not in seen]
            verdict = replace(verdict, recoverable=bool(options) and not verdict.passed)
            event = {
                "task_id": self.task_id, "step_id": current.get("step_id"),
                "planned_action": current, "tool_name": current.get("tool"),
                "tool_arguments": current.get("arguments", {}),
                "tool_result_status": status, "verification_passed": verdict.passed,
                "failure_type": verdict.failure_type, "failure_reason": verdict.reason,
                "evidence": verdict.evidence, "scope": verdict.scope, "recoverable": verdict.recoverable,
                "reflection_triggered": False, "repair_action": None, "retry_index": retry_index,
                "final_status": "passed" if verdict.passed else "failed",
                "latency_seconds": time.perf_counter() - started,
                "action_hash": hashlib.sha256(fingerprint.encode()).hexdigest(),
            }
            if verdict.passed:
                event["final_status"] = "recovered" if retry_index else "passed"
                self._write(event)
                return value
            enabled = self.config.get("reflection", {}).get("enabled", False)
            if not (enabled and verdict.recoverable and self.used_retries < self.max_retries):
                event["termination_reason"] = ("retry_budget_exhausted" if enabled and verdict.recoverable
                                               else "reflection_disabled_or_no_supported_repair")
                self._write(event)
                raise VerifiedExecutionError(verdict, str(self.path))
            event["reflection_triggered"] = True
            try:
                if self.reflector is None:
                    from ..llm.deepseek import DeepSeekClient
                    from .reflection import LLMReflector
                    llm = self.config.get("llm", {})
                    self.reflector = LLMReflector(
                        DeepSeekClient(base_url=llm.get("base_url", "https://api.deepseek.com")),
                        llm.get("parameter_model", "deepseek-v4-flash"),
                    )
                response = self.reflector.reflect(serializable({
                    "original_task": self.task, "current_plan": self.task.get("plan"),
                    "failed_action": current, **verdict.to_dict(),
                    "previous_execution_trace": self.history,
                    "allowed_repairs": options,
                }))
                selected = next(option for option in options if option["id"] == response["repair_id"])
                event["reflection"] = response
                event["repair_action"] = selected["action"]
                event["final_status"] = "repair_pending"
            except Exception as exc:
                event["reflection_error"] = f"{type(exc).__name__}: {exc}"
                self._write(event)
                raise VerifiedExecutionError(verdict, str(self.path)) from exc
            self.used_retries += 1
            self._write(event)
            current = copy.deepcopy(selected["action"])
            retry_index += 1

    def finish(self, status, error=None):
        self._write({"task_id": self.task_id, "step_id": None, "planned_action": None,
                     "tool_name": None, "tool_arguments": {}, "tool_result_status": "not_called",
                     "verification_passed": status == "success", "failure_type": None,
                     "failure_reason": error, "reflection_triggered": False, "repair_action": None,
                     "retry_index": self.used_retries, "final_status": status, "event": "task_end"})
