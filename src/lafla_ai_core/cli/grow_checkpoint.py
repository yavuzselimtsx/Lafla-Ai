"""
@Dosya: cli/grow_checkpoint.py
@Aciklama: 100M Lafla checkpointini ayni genislikteki 200M depth-growth baslangicina cevirir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.model.growth import build_depth_growth_plan, grow_state_dict


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla checkpoint depth growth")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-model-config", default="configs/model/lafla-200m-thinking.yaml")
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args(argv)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("checkpoint growth icin torch kurulu olmali") from exc

    from lafla_ai_core.model.checkpoint_contract import validate_checkpoint_directory
    from lafla_ai_core.model.checkpoint_io import save_training_checkpoint
    from lafla_ai_core.model.transformer import LaflaDecoderModel

    source_dir = Path(args.source_checkpoint)
    validate_checkpoint_directory(source_dir)
    source_payload = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    source_config = ModelConfig.from_mapping(source_payload)
    target_config = ModelConfig.from_mapping(load_mapping(args.target_model_config))
    plan = build_depth_growth_plan(source_config, target_config)

    source_state = torch.load(source_dir / "model.pt", map_location="cpu")
    target_model = LaflaDecoderModel(target_config)
    target_state = grow_state_dict(source_state, target_model.state_dict(), source_config, target_config)
    target_model.load_state_dict(target_state, strict=True)
    optimizer = torch.optim.AdamW(_optimizer_groups(target_model), lr=1e-4)
    output = Path(args.output_checkpoint)
    save_training_checkpoint(
        output,
        target_model,
        optimizer,
        target_config,
        {
            "step": 0,
            "cumulative_tokens": 0,
            "growth": plan.to_mapping(),
            "continued_pretraining_required": True,
            "format": "lafla-trainer-state-v2",
        },
    )
    (output / "growth-report.json").write_text(
        json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _optimizer_groups(model: object) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():  # type: ignore[attr-defined]
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "norm" in name.lower() or name.lower().endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


if __name__ == "__main__":
    raise SystemExit(main())
