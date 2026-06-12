"""
@Dosya: training/runner.py
@Aciklama: LaflaAi-Core icin gercek next-token pretraining kosucusu.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab T4 zamanini korumak icin resume, health log, checkpoint
        retention ve smoke kosusu ayni kod yolunda tutulur.
@Uyari: Bu modul production fallback uydurmaz; eksik bagimlilik veya veri hatasi
        egitimi baslatmadan durdurur.
@Calisma-Semasi: configs -> model/data -> train loop -> checkpoints/reports
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("lafla_ai_core.training.runner icin torch kurulu olmali") from exc

from lafla_ai_core.config.schema import ModelConfig, TrainingConfig
from lafla_ai_core.data.packing import TokenizersCodec, iter_jsonl_texts, iter_packed_token_blocks, resolve_special_token_id
from lafla_ai_core.data.routing import assert_pretraining_inputs
from lafla_ai_core.model.checkpoint_io import load_training_checkpoint, save_training_checkpoint
from lafla_ai_core.model.transformer import LaflaDecoderModel
from lafla_ai_core.training.checkpoint_policy import CheckpointPolicy, retention_victims, should_save_checkpoint
from lafla_ai_core.training.curriculum import CurriculumStage, resolve_curriculum_stage, tokens_per_optimizer_step
from lafla_ai_core.training.lr_schedule import cosine_with_warmup_lr
from lafla_ai_core.training.parallelism import (
    ParallelismDecision,
    resolve_batch_geometry,
    resolve_data_parallel,
    resolve_gradient_checkpointing,
)
from lafla_ai_core.training.stability import StabilityMonitor


@dataclass(frozen=True)
class TrainingPaths:
    """Egitim dosya yollarini tasir."""

    data_jsonl: tuple[str, ...]
    tokenizer_path: str
    checkpoint_dir: str
    health_log_path: str
    resume_from: str | None = None


@dataclass(frozen=True)
class TrainingSummary:
    """Egitim tamamlaninca donen ozet."""

    final_step: int
    final_checkpoint: str
    health_log_path: str
    seconds: float
    cumulative_tokens: int


class JsonlTokenBlockDataset(IterableDataset[torch.Tensor]):
    """JSONL kaynaklardan sonsuz token blok akisi uretir."""

    def __init__(self, paths: Sequence[str], tokenizer_path: str, sequence_length: int, eos_id: int) -> None:
        super().__init__()
        self.paths = tuple(paths)
        self.tokenizer_path = tokenizer_path
        self.sequence_length = sequence_length
        self.eos_id = eos_id

    def __iter__(self) -> Iterable[torch.Tensor]:
        tokenizer = TokenizersCodec(self.tokenizer_path)
        while True:
            for block in iter_packed_token_blocks(iter_jsonl_texts(self.paths), tokenizer, self.sequence_length, self.eos_id):
                yield torch.tensor(block, dtype=torch.long)


class SmokeTokenBlockDataset(IterableDataset[torch.Tensor]):
    """Ayni training yolunu hizli denemek icin sentetik token bloklari."""

    def __init__(self, vocab_size: int, sequence_length: int, seed: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.seed = seed

    def __iter__(self) -> Iterable[torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            yield torch.randint(0, self.vocab_size, (self.sequence_length,), generator=generator, dtype=torch.long)


class _TrainingLossModule(torch.nn.Module):
    """DataParallel'in rahat toplayacagi skaler loss arayuzu."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("loss uretilemedi")
        return output.loss


def run_pretraining(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    paths: TrainingPaths,
    *,
    smoke: bool = False,
) -> TrainingSummary:
    """Pretraining kosusunu baslatir ve final checkpoint yazar."""

    model_config.validate()
    training_config.validate()
    _set_seed(training_config.seed)
    device, xla_sync = _select_training_device(training_config)
    dtype = _autocast_dtype(training_config.precision)
    checkpoint_root = Path(paths.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    health_log = Path(paths.health_log_path)
    health_log.parent.mkdir(parents=True, exist_ok=True)
    _ensure_free_space(checkpoint_root, training_config.checkpoint_min_free_gb)

    eos_id = 0
    tokenizer_vocab_size = None
    if not smoke:
        if not paths.data_jsonl:
            raise ValueError("gercek egitim icin en az bir data_jsonl gerekli")
        assert_pretraining_inputs(paths.data_jsonl)
        tokenizer_probe = TokenizersCodec(paths.tokenizer_path)
        eos_id = resolve_special_token_id(tokenizer_probe, "<|eos|>")
        tokenizer_vocab_size = tokenizer_probe.vocab_size()
        if tokenizer_vocab_size != model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size model vocab_size ile ayni olmali: tokenizer={tokenizer_vocab_size}, model={model_config.vocab_size}"
            )

    base_model = LaflaDecoderModel(model_config).to(device)
    train_model = _TrainingLossModule(base_model)
    parallel_decision = _resolve_data_parallel(training_config, device)
    batch_geometry = resolve_batch_geometry(
        configured_micro_batch_size=training_config.micro_batch_size,
        configured_gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        cuda_micro_batch_size_per_device=training_config.cuda_micro_batch_size_per_device,
        target_sequences_per_optimizer_step=training_config.target_sequences_per_optimizer_step,
        decision=parallel_decision,
    )
    active_micro_batch_size = batch_geometry.global_micro_batch_size
    active_gradient_accumulation_steps = batch_geometry.gradient_accumulation_steps
    if parallel_decision.enabled:
        train_model = torch.nn.DataParallel(train_model)
    optimizer = build_optimizer(train_model, training_config)
    start_step = 0
    cumulative_tokens = 0
    if paths.resume_from:
        state = load_training_checkpoint(paths.resume_from, base_model, optimizer, map_location="cpu" if device.type == "xla" else device)
        _move_optimizer_state(optimizer, device)
        start_step = int(state.get("step", 0))
        cumulative_tokens = int(state.get("cumulative_tokens", 0))

    active_stage = resolve_curriculum_stage(training_config, cumulative_tokens)
    active_gradient_checkpointing = resolve_gradient_checkpointing(
        model_checkpointing_enabled=model_config.gradient_checkpointing,
        minimum_sequence_length=training_config.gradient_checkpointing_min_sequence_length,
        active_sequence_length=active_stage.sequence_length,
    )
    base_model.set_gradient_checkpointing(active_gradient_checkpointing)
    iterator = _build_training_iterator(
        model_config,
        training_config,
        paths,
        active_stage,
        eos_id=eos_id,
        smoke=smoke,
        micro_batch_size=active_micro_batch_size,
    )
    scaler = _build_grad_scaler(device, training_config.precision)
    policy = CheckpointPolicy(training_config.save_every, training_config.keep_last_checkpoints)
    stability = StabilityMonitor()
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    if start_step > 0:
        _append_health(
            health_log,
            {
                "step": start_step,
                "event": "resume",
                "resume_from": paths.resume_from,
                "dataset_position_policy": "restart_stream_without_slow_skip",
                "cumulative_tokens": cumulative_tokens,
                "curriculum_stage": active_stage.index,
                "sequence_length": active_stage.sequence_length,
                "parallelism": parallel_decision.mode,
                "cuda_device_count": parallel_decision.cuda_device_count,
                "configured_micro_batch_size": training_config.micro_batch_size,
                "effective_micro_batch_size": active_micro_batch_size,
                "configured_gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "effective_gradient_accumulation_steps": active_gradient_accumulation_steps,
                "sequences_per_optimizer_step": batch_geometry.sequences_per_optimizer_step,
                "gradient_checkpointing": active_gradient_checkpointing,
            },
        )

    last_completed_step = start_step
    try:
        for step in range(start_step + 1, training_config.max_steps + 1):
            if training_config.target_tokens and cumulative_tokens >= training_config.target_tokens:
                break
            stage = resolve_curriculum_stage(training_config, cumulative_tokens)
            if stage.index != active_stage.index:
                active_stage = stage
                active_gradient_checkpointing = resolve_gradient_checkpointing(
                    model_checkpointing_enabled=model_config.gradient_checkpointing,
                    minimum_sequence_length=training_config.gradient_checkpointing_min_sequence_length,
                    active_sequence_length=active_stage.sequence_length,
                )
                base_model.set_gradient_checkpointing(active_gradient_checkpointing)
                iterator = _build_training_iterator(
                    model_config,
                    training_config,
                    paths,
                    active_stage,
                    eos_id=eos_id,
                    smoke=smoke,
                    micro_batch_size=active_micro_batch_size,
                )
                _append_health(
                    health_log,
                    {
                        "step": step,
                        "event": "curriculum_transition",
                        "cumulative_tokens": cumulative_tokens,
                        "curriculum_stage": active_stage.index,
                        "sequence_length": active_stage.sequence_length,
                        "configured_micro_batch_size": training_config.micro_batch_size,
                        "effective_micro_batch_size": active_micro_batch_size,
                        "configured_gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                        "effective_gradient_accumulation_steps": active_gradient_accumulation_steps,
                        "sequences_per_optimizer_step": batch_geometry.sequences_per_optimizer_step,
                        "gradient_checkpointing": active_gradient_checkpointing,
                    },
                )
            lr = cosine_with_warmup_lr(
                step - 1,
                training_config.max_steps,
                training_config.warmup_steps,
                training_config.learning_rate,
                training_config.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            accumulated_loss = 0.0
            for micro_step in range(active_gradient_accumulation_steps):
                batch = next(iterator).to(device)
                labels = batch.clone()
                with _autocast_context(device, dtype):
                    raw_loss = train_model(batch, labels=labels)
                    step_loss = raw_loss.mean()
                    if not torch.isfinite(step_loss.detach()):
                        raise RuntimeError(
                            f"finite olmayan loss: step={step}, micro_step={micro_step}, loss={float(step_loss.detach().cpu())}"
                        )
                    loss = step_loss / active_gradient_accumulation_steps
                accumulated_loss += float(step_loss.detach().cpu())
                scaler.scale(loss).backward()
                if xla_sync is not None:
                    xla_sync()
            scaler.unscale_(optimizer)
            # XLA dahil tum cihazlarda clipping uygulanir; grad_clip_norm config
            # degeri hicbir accelerator yolunda sessizce yok sayilmamali.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(train_model.parameters(), training_config.grad_clip_norm))
            if xla_sync is None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                xla_sync()
            optimizer.zero_grad(set_to_none=True)

            average_loss = accumulated_loss / active_gradient_accumulation_steps
            previous_tokens = cumulative_tokens
            cumulative_tokens += tokens_per_optimizer_step(
                training_config,
                active_stage,
                micro_batch_size=active_micro_batch_size,
                gradient_accumulation_steps=active_gradient_accumulation_steps,
            )
            stability_observation = stability.observe(step=step, loss=average_loss, grad_norm=grad_norm)
            if step % training_config.log_every == 0 or step == 1:
                _append_health(
                    health_log,
                    {
                        "step": step,
                        "loss": round(average_loss, 6),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "loss_spike": stability_observation.loss_spike,
                        "grad_norm_spike": stability_observation.grad_norm_spike,
                        "spike_score": round(stability_observation.spike_score, 6),
                        "device": str(device),
                        "accelerator": training_config.accelerator,
                        "parallelism": parallel_decision.mode,
                        "data_parallel": parallel_decision.enabled,
                        "data_parallel_reason": parallel_decision.reason,
                        "cuda_device_count": parallel_decision.cuda_device_count,
                        "configured_micro_batch_size": training_config.micro_batch_size,
                        "effective_micro_batch_size": active_micro_batch_size,
                        "configured_gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                        "effective_gradient_accumulation_steps": active_gradient_accumulation_steps,
                        "sequences_per_optimizer_step": batch_geometry.sequences_per_optimizer_step,
                        "gradient_checkpointing": active_gradient_checkpointing,
                        "tokenizer_vocab_size": tokenizer_vocab_size,
                        "cumulative_tokens": cumulative_tokens,
                        "curriculum_stage": active_stage.index,
                        "sequence_length": active_stage.sequence_length,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                )
            token_checkpoint_due = bool(
                training_config.checkpoint_every_tokens
                and previous_tokens // training_config.checkpoint_every_tokens
                < cumulative_tokens // training_config.checkpoint_every_tokens
            )
            if should_save_checkpoint(step, training_config.max_steps, policy) or token_checkpoint_due:
                save_training_checkpoint(
                    checkpoint_root / f"lafla-step-{step:06d}",
                    base_model,
                    optimizer,
                    model_config,
                    _trainer_state(step, training_config, smoke, cumulative_tokens, active_stage),
                )
                _apply_retention(checkpoint_root, training_config.keep_last_checkpoints)
            last_completed_step = step
    except KeyboardInterrupt:
        if last_completed_step > start_step:
            save_training_checkpoint(
                checkpoint_root / f"lafla-interrupted-step-{last_completed_step:06d}",
                base_model,
                optimizer,
                model_config,
                _trainer_state(last_completed_step, training_config, smoke, cumulative_tokens, active_stage),
            )
        raise

    final_dir = checkpoint_root / "lafla-final"
    save_training_checkpoint(
        final_dir,
        base_model,
        optimizer,
        model_config,
        _trainer_state(last_completed_step, training_config, smoke, cumulative_tokens, active_stage),
    )
    return TrainingSummary(
        final_step=last_completed_step,
        final_checkpoint=str(final_dir),
        health_log_path=str(health_log),
        seconds=time.time() - started,
        cumulative_tokens=cumulative_tokens,
    )


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Config'e gore optimizer kurar."""

    param_groups = _optimizer_param_groups(model, config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.learning_rate)
    if config.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("adamw8bit icin bitsandbytes gerekli; Colab'da requirements kurulumunu calistirin") from exc
        return bnb.optim.AdamW8bit(param_groups, lr=config.learning_rate)
    if config.optimizer == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lion optimizer icin lion-pytorch gerekli veya optimizer=adamw secilmeli") from exc
        return Lion(param_groups, lr=config.learning_rate)
    raise ValueError(f"desteklenmeyen optimizer: {config.optimizer}")


def _trainer_state(
    step: int,
    config: TrainingConfig,
    smoke: bool,
    cumulative_tokens: int,
    stage: CurriculumStage,
) -> dict[str, object]:
    return {
        "step": step,
        "cumulative_tokens": cumulative_tokens,
        "curriculum_stage": stage.index,
        "sequence_length": stage.sequence_length,
        "training": asdict(config),
        "smoke": smoke,
        "format": "lafla-trainer-state-v2",
    }


def _build_training_iterator(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    paths: TrainingPaths,
    stage: CurriculumStage,
    *,
    eos_id: int,
    smoke: bool,
    micro_batch_size: int | None = None,
):
    dataset: IterableDataset[torch.Tensor]
    if smoke:
        dataset = SmokeTokenBlockDataset(model_config.vocab_size, stage.sequence_length, training_config.seed + stage.index)
    else:
        dataset = JsonlTokenBlockDataset(paths.data_jsonl, paths.tokenizer_path, stage.sequence_length, eos_id=eos_id)
    loader = DataLoader(dataset, batch_size=training_config.micro_batch_size if micro_batch_size is None else micro_batch_size)
    return iter(loader)


def _append_health(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if parameter.ndim < 2 or "norm" in lowered or lowered.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("optimizer icin egitilebilir parametre yok")
    return groups


def _apply_retention(checkpoint_root: Path, keep_last: int) -> None:
    steps: list[int] = []
    for child in checkpoint_root.glob("lafla-step-*"):
        if child.is_dir() and (child / "READY.json").exists():
            try:
                steps.append(int(child.name.rsplit("-", 1)[-1]))
            except ValueError:
                continue
    for victim in retention_victims(steps, keep_last):
        target = checkpoint_root / f"lafla-step-{victim:06d}"
        resolved_root = checkpoint_root.resolve()
        resolved_target = target.resolve()
        if resolved_root not in resolved_target.parents:
            raise RuntimeError(f"retention hedefi checkpoint disinda: {resolved_target}")
        shutil.rmtree(resolved_target)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_training_device(config: TrainingConfig) -> tuple[torch.device, Callable[[], None] | None]:
    accelerator = config.accelerator
    if accelerator == "xla":
        return _xla_device()
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("accelerator=cuda secildi ama CUDA kullanilabilir degil")
        return torch.device("cuda"), None
    if accelerator == "cpu":
        return torch.device("cpu"), None
    if _looks_like_tpu_runtime():
        try:
            return _xla_device()
        except ModuleNotFoundError:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda"), None
    return torch.device("cpu"), None


def _resolve_data_parallel(config: TrainingConfig, device: torch.device) -> ParallelismDecision:
    """CUDA coklu GPU kararini verir; tek cihaz/CPU/TPU durumunda sessizce tek cihaza duser."""

    cuda_device_count = torch.cuda.device_count() if device.type == "cuda" else 0
    return resolve_data_parallel(config.data_parallel, device.type, cuda_device_count)


def _looks_like_tpu_runtime() -> bool:
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or bool(os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME"))


def _xla_device() -> tuple[torch.device, Callable[[], None]]:
    try:
        import torch_xla  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TPU/XLA egitimi icin `pip install torch torch_xla[tpu]` gerekli") from exc
    if hasattr(torch_xla, "device") and hasattr(torch_xla, "sync"):
        return torch_xla.device(), torch_xla.sync  # type: ignore[no-any-return]
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch_xla kurulu ama xla_model API bulunamadi") from exc
    return xm.xla_device(), xm.mark_step


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return None
    raise ValueError(f"desteklenmeyen precision: {precision}")


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "xla" and dtype is torch.bfloat16:
        # torch_xla autocast entegrasyonu: precision=bf16 TPU'da da etkin olmali,
        # aksi halde config sessizce yok sayilip fp32 calisirdi.
        try:
            return torch.autocast(device_type="xla", dtype=torch.bfloat16)
        except RuntimeError:  # pragma: no cover - eski torch_xla surumleri
            return nullcontext()
    return nullcontext()


class _NoopGradScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _build_grad_scaler(device: torch.device, precision: str):
    if device.type != "cuda":
        return _NoopGradScaler()
    enabled = device.type == "cuda" and precision == "fp16"
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _ensure_free_space(path: Path, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"checkpoint diski yetersiz: {free_gb:.2f} GB bos, gereken {min_free_gb:.2f} GB")
