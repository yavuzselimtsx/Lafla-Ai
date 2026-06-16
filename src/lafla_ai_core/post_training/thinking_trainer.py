"""
@Dosya: post_training/thinking_trainer.py
@Aciklama: Thinking SFT kayitlarini checkpoint uzerinde kisa supervised fine-tune kosusuna cevirir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu kosucu base pretrain checkpointini degistirmez; cikisi ayri checkpoint
        klasorune yazar.
@Uyari: Tokenizer degistirilmez. Kaynak checkpoint ve SFT cikisi ayri tutulmalidir.
@Calisma-Semasi: thinking jsonl -> supervised labels -> train loop -> SFT checkpoint
"""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError:  # pragma: no cover - helper fonksiyonlar torch olmadan test edilebilir
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

    class IterableDataset:  # type: ignore[no-redef]
        pass

from lafla_ai_core.config.schema import ModelConfig, PostTrainingConfig
from lafla_ai_core.data.packing import TokenizersCodec, resolve_special_token_id
from lafla_ai_core.post_training.thinking_dataset import (
    iter_thinking_jsonl_records,
    validate_thinking_jsonl_file,
)
from lafla_ai_core.post_training.thinking_sft import (
    THINK_CLOSE,
    THINK_OPEN,
    ChatTurn,
    SupervisedChatExample,
    build_supervised_chat_example,
)


@dataclass(frozen=True)
class PaddedSftExample:
    """Sabit uzunluklu SFT tensore donusmeden onceki ornek."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class ThinkingSftTrainingPaths:
    """Thinking SFT kosusu icin dosya yollarini tasir."""

    source_checkpoint: str
    tokenizer_path: str
    data_jsonl: tuple[str, ...]
    output_dir: str
    health_log_path: str


@dataclass(frozen=True)
class ThinkingSftTrainingSummary:
    """Thinking SFT tamamlaninca basilan ozet."""

    source_checkpoint: str
    output_checkpoint: str
    health_log_path: str
    optimizer_steps: int
    micro_steps: int
    total_examples: int
    tokens_seen: int
    seconds: float


class ThinkingSftIterableDataset(IterableDataset):
    """SFT orneklerini bellekte biriktirmeden epoch bazli akar."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        tokenizer_path: str | Path,
        config: PostTrainingConfig,
        *,
        pad_id: int,
        seed: int,
        shuffle_buffer_size: int,
    ) -> None:
        super().__init__()
        self.paths = tuple(str(path) for path in paths)
        self.tokenizer_path = str(tokenizer_path)
        self.config = config
        self.pad_id = pad_id
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterable[tuple[object, object]]:
        torch_module = _require_torch()
        tokenizer = TokenizersCodec(self.tokenizer_path)
        for _epoch in range(self.config.epochs):
            examples = iter_supervised_thinking_examples(
                self.paths,
                tokenizer,
                self.config,
                validate_inputs=False,
            )
            examples = iter_buffer_shuffled(
                examples,
                seed=self.seed + _epoch,
                buffer_size=self.shuffle_buffer_size,
            )
            for example in examples:
                padded = pad_or_truncate_example(
                    example.input_ids,
                    example.labels,
                    sequence_length=self.config.sequence_length,
                    pad_id=self.pad_id,
                )
                if not _has_trainable_shift_label(padded.labels):
                    continue
                yield (
                    torch_module.tensor(padded.input_ids, dtype=torch_module.long),
                    torch_module.tensor(padded.labels, dtype=torch_module.long),
                )


def iter_supervised_thinking_examples(
    paths: Sequence[str | Path],
    tokenizer: object,
    config: PostTrainingConfig,
    *,
    validate_inputs: bool = True,
) -> Iterable[SupervisedChatExample]:
    """Thinking JSONL kayitlarini supervised chat orneklerine cevirir."""

    config.validate()
    if validate_inputs:
        _validate_data_files(paths, config)
    supervise_thinking = config.label_policy == "assistant_with_thinking"
    for path in paths:
        for _line_number, record in iter_thinking_jsonl_records(path):
            assistant = f"{THINK_OPEN}{record.thinking.strip()}{THINK_CLOSE}\n{record.assistant.strip()}"
            turns = (
                ChatTurn("system", record.system),
                ChatTurn("user", record.user),
                ChatTurn("assistant", assistant),
            )
            yield build_supervised_chat_example(
                turns,
                tokenizer,
                only_last_assistant=config.only_last_assistant,
                supervise_thinking=supervise_thinking,
            )


def iter_buffer_shuffled(
    items: Iterable[object],
    *,
    seed: int,
    buffer_size: int,
) -> Iterable[object]:
    """Buyuk dosyayi bellekte tamamen tutmadan deterministik karistirir."""

    if buffer_size <= 1:
        yield from items
        return
    rng = random.Random(seed)
    buffer: list[object] = []
    for item in items:
        buffer.append(item)
        if len(buffer) < buffer_size:
            continue
        selected = rng.randrange(len(buffer))
        yield buffer.pop(selected)
    while buffer:
        selected = rng.randrange(len(buffer))
        yield buffer.pop(selected)


def pad_or_truncate_example(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    sequence_length: int,
    pad_id: int,
) -> PaddedSftExample:
    """Ornegi sabit uzunluga getirir; uzun kayitta cevap kuyrugunu korur."""

    if sequence_length < 2:
        raise ValueError("sequence_length en az 2 olmali")
    if len(input_ids) != len(labels):
        raise ValueError("input_ids ve labels ayni uzunlukta olmali")
    ids = tuple(int(token) for token in input_ids)
    target_labels = tuple(int(token) for token in labels)
    if len(ids) > sequence_length:
        ids = ids[-sequence_length:]
        target_labels = target_labels[-sequence_length:]
    pad_count = sequence_length - len(ids)
    if pad_count > 0:
        ids = ids + (int(pad_id),) * pad_count
        target_labels = target_labels + (-100,) * pad_count
    return PaddedSftExample(input_ids=ids, labels=target_labels)


def run_thinking_sft(
    config: PostTrainingConfig,
    paths: ThinkingSftTrainingPaths,
    *,
    device: str = "auto",
    precision: str = "fp16",
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int = 0,
    weight_decay: float = 0.0,
    grad_clip_norm: float = 1.0,
    seed: int = 1337,
    shuffle_buffer_size: int = 4096,
) -> ThinkingSftTrainingSummary:
    """Kaynak checkpointten ayri bir thinking SFT checkpoint uretir."""

    torch_module = _require_torch()
    config.validate()
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size pozitif olmali")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps pozitif olmali")
    if max_steps < 0:
        raise ValueError("max_steps negatif olamaz")
    if shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size negatif olamaz")
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("desteklenmeyen precision")
    _validate_data_files(paths.data_jsonl, config)
    _set_seed(seed)

    selected_device = _select_device(device)
    from lafla_ai_core.model.checkpoint_io import load_training_checkpoint, save_training_checkpoint
    from lafla_ai_core.model.transformer import LaflaDecoderModel

    model_config = ModelConfig.from_mapping(
        json.loads((Path(paths.source_checkpoint) / "config.json").read_text(encoding="utf-8"))
    )
    model_config.validate()
    tokenizer = TokenizersCodec(paths.tokenizer_path)
    pad_id = resolve_special_token_id(tokenizer, "<|pad|>")
    if tokenizer.vocab_size() != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size model vocab_size ile ayni olmali: tokenizer={tokenizer.vocab_size()}, model={model_config.vocab_size}"
        )

    model = LaflaDecoderModel(model_config).to(selected_device)
    model.set_gradient_checkpointing(False)
    load_training_checkpoint(paths.source_checkpoint, model, optimizer=None, map_location="cpu")
    optimizer = torch_module.optim.AdamW(
        _optimizer_param_groups(model, weight_decay),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )
    dataset = ThinkingSftIterableDataset(
        paths.data_jsonl,
        paths.tokenizer_path,
        config,
        pad_id=pad_id,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    if DataLoader is None:
        raise ModuleNotFoundError("thinking SFT egitimi icin torch kurulu olmali")
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        pin_memory=selected_device.type == "cuda",
    )
    dtype = _autocast_dtype(precision)
    scaler = _build_grad_scaler(selected_device, precision)
    health_log = Path(paths.health_log_path)
    health_log.parent.mkdir(parents=True, exist_ok=True)
    Path(paths.output_dir).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    optimizer_steps = 0
    micro_steps = 0
    total_examples = 0
    tokens_seen = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for input_ids, labels in loader:
        input_ids = input_ids.to(selected_device, non_blocking=selected_device.type == "cuda")
        labels = labels.to(selected_device, non_blocking=selected_device.type == "cuda")
        with _autocast_context(selected_device, dtype):
            output = model(input_ids, labels=labels)
            if output.loss is None:
                raise RuntimeError("SFT loss uretilemedi")
            step_loss = output.loss
            if not torch_module.isfinite(step_loss.detach()):
                raise RuntimeError(f"finite olmayan SFT loss: {float(step_loss.detach().cpu())}")
            loss = step_loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        accumulated_loss += float(step_loss.detach().cpu())
        micro_steps += 1
        total_examples += int(input_ids.shape[0])
        tokens_seen += int(input_ids.numel())
        if micro_steps % gradient_accumulation_steps != 0:
            continue
        scaler.unscale_(optimizer)
        grad_norm = float(torch_module.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        _append_health(
            health_log,
            {
                "stage": config.stage,
                "step": optimizer_steps,
                "micro_steps": micro_steps,
                "loss": round(accumulated_loss / gradient_accumulation_steps, 6),
                "lr": config.learning_rate,
                "grad_norm": grad_norm,
                "device": str(selected_device),
                "precision": precision,
                "sequence_length": config.sequence_length,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "shuffle_buffer_size": shuffle_buffer_size,
                "total_examples": total_examples,
                "tokens_seen": tokens_seen,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        accumulated_loss = 0.0
        if max_steps and optimizer_steps >= max_steps:
            break

    if optimizer_steps == 0:
        raise RuntimeError("SFT hic optimizer step tamamlamadi")
    save_training_checkpoint(
        paths.output_dir,
        model,
        optimizer,
        model_config,
        build_sft_trainer_state(
            step=optimizer_steps,
            epoch=config.epochs,
            source_checkpoint=paths.source_checkpoint,
            total_examples=total_examples,
            tokens_seen=tokens_seen,
            config=config,
        ),
    )
    return ThinkingSftTrainingSummary(
        source_checkpoint=paths.source_checkpoint,
        output_checkpoint=paths.output_dir,
        health_log_path=paths.health_log_path,
        optimizer_steps=optimizer_steps,
        micro_steps=micro_steps,
        total_examples=total_examples,
        tokens_seen=tokens_seen,
        seconds=time.time() - started,
    )


def build_sft_trainer_state(
    *,
    step: int,
    epoch: int,
    source_checkpoint: str,
    total_examples: int,
    tokens_seen: int,
    config: PostTrainingConfig,
) -> dict[str, object]:
    """SFT checkpoint trainer_state payload'unu uretir."""

    return {
        "step": int(step),
        "epoch": int(epoch),
        "source_checkpoint": str(source_checkpoint),
        "total_examples": int(total_examples),
        "tokens_seen": int(tokens_seen),
        "post_training": asdict(config),
        "format": "lafla-thinking-sft-state-v1",
    }


def _validate_data_files(paths: Sequence[str | Path], config: PostTrainingConfig) -> None:
    if not paths:
        raise ValueError("thinking SFT icin en az bir data_jsonl gerekli")
    for path in paths:
        report = validate_thinking_jsonl_file(path, max_thinking_chars=config.max_thinking_chars)
        if not report.ok:
            details = "; ".join(f"{finding.path}:{finding.line}:{finding.code}" for finding in report.findings[:10])
            raise ValueError(f"thinking SFT veri kapisi basarisiz: {details}")


def _has_trainable_shift_label(labels: Sequence[int]) -> bool:
    return any(int(label) != -100 for label in labels[1:])


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
        raise ValueError("SFT optimizer icin egitilebilir parametre yok")
    return groups


def _set_seed(seed: int) -> None:
    torch_module = _require_torch()
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _select_device(device: str) -> torch.device:
    torch_module = _require_torch()
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("device=cuda secildi ama CUDA kullanilabilir degil")
    if device not in {"cpu", "cuda"}:
        raise ValueError("SFT icin device auto/cpu/cuda olmali")
    return torch_module.device(device)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    torch_module = _require_torch()
    if precision == "fp16":
        return torch_module.float16
    if precision == "bf16":
        return torch_module.bfloat16
    if precision == "fp32":
        return None
    raise ValueError(f"desteklenmeyen precision: {precision}")


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    torch_module = _require_torch()
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=dtype)


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
    torch_module = _require_torch()
    if device.type != "cuda" or precision != "fp16":
        return _NoopGradScaler()
    amp = getattr(torch_module, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp.GradScaler(enabled=True)
    return torch_module.cuda.amp.GradScaler(enabled=True)


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("thinking SFT egitimi icin torch kurulu olmali")
    return torch


def _append_health(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
