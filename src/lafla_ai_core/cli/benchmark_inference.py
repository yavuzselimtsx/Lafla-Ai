"""
@Dosya: cli/benchmark_inference.py
@Aciklama: HF inference process-tree peak RSS profilini JSON olarak olcer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig, RuntimeConfig
from lafla_ai_core.runtime.memory_budget import MIB, estimate_runtime_memory
from lafla_ai_core.runtime.rss import monitor_command


DEFAULT_PROFILES = (2048, 8192, 15360, 20000)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla gercek inference RSS benchmark")
    parser.add_argument("--model-config", default="configs/model/lafla-100m-thinking.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/desktop-i3-int8-100m.yaml")
    parser.add_argument("--model-dir")
    parser.add_argument("--output", default="artifacts/benchmarks/lafla-100m-rss.json")
    parser.add_argument("--profiles", nargs="*", type=int, default=list(DEFAULT_PROFILES))
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--context-tokens", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _run_worker(args.model_dir, args.context_tokens)

    model = ModelConfig.from_mapping(load_mapping(args.model_config))
    runtime = RuntimeConfig.from_mapping(load_mapping(args.runtime_config))
    model.validate()
    runtime.validate()
    estimate = estimate_runtime_memory(model, runtime)
    payload: dict[str, object] = {
        "status": "blocked",
        "accepted": False,
        "model_config": args.model_config,
        "runtime_config": args.runtime_config,
        "estimate": asdict(estimate),
        "profiles": [],
        "acceptance_limit_mib": runtime.peak_rss_limit_mib,
    }
    missing = [
        package
        for package in ("psutil", "torch", "transformers")
        if importlib.util.find_spec(package) is None
    ]
    model_dir = Path(args.model_dir) if args.model_dir else None
    if missing:
        payload["reason"] = f"eksik dependency: {', '.join(missing)}"
        _write_report(args.output, payload)
        return 2
    if model_dir is None or not (model_dir / "config.json").exists():
        payload["reason"] = "gercek HF model klasoru ve config.json zorunlu"
        _write_report(args.output, payload)
        return 2

    profile_results: list[dict[str, object]] = []
    accepted = True
    for context_tokens in args.profiles:
        if not 1 <= context_tokens <= runtime.context_length:
            profile_results.append({"context_tokens": context_tokens, "status": "invalid"})
            accepted = False
            continue
        command = (
            sys.executable,
            "-m",
            "lafla_ai_core.cli.benchmark_inference",
            "--worker",
            "--model-dir",
            str(model_dir),
            "--context-tokens",
            str(context_tokens),
        )
        peak = monitor_command(command, timeout_seconds=args.timeout_seconds)
        profile_ok = peak.return_code == 0 and peak.peak_rss_bytes <= runtime.peak_rss_limit_mib * MIB
        accepted = accepted and profile_ok
        profile_results.append(
            {
                "context_tokens": context_tokens,
                "status": "passed" if profile_ok else "failed",
                **asdict(peak),
                "peak_rss_mib": peak.peak_rss_bytes / MIB,
                "peak_uss_mib": peak.peak_uss_bytes / MIB,
            }
        )
    payload["profiles"] = profile_results
    payload["accepted"] = accepted
    payload["status"] = "passed" if accepted else "failed"
    _write_report(args.output, payload)
    return 0 if accepted else 2


def _run_worker(model_dir: str | None, context_tokens: int | None) -> int:
    if model_dir is None or context_tokens is None:
        return 2
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    token_id = int(model.config.bos_token_id or 0)
    input_ids = torch.full((1, context_tokens), token_id, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            num_logits_to_keep=1,
        )
        if context_tokens < int(model.config.context_length):
            next_ids = torch.full((1, 1), token_id, dtype=torch.long)
            model(
                input_ids=next_ids,
                attention_mask=torch.ones((1, context_tokens + 1), dtype=torch.long),
                past_key_values=output.past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
    return 0


def _write_report(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
