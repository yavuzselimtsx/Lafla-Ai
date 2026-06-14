"""
@Dosya: runtime/checkpoint_inference.py
@Aciklama: Egitim checkpoint'ini yukleyip Lafla chat promptu ile smoke generation kosar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab 100-step checkpoint testi notebook icine gomulmez; ayni kod yolu
        CLI ve operator komutlari tarafindan kullanilir.
@Uyari: Bu modul torch'u yalniz gercek generation kosusunda import eder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.schema import ModelConfig, RuntimeConfig
from lafla_ai_core.model.checkpoint_contract import validate_checkpoint_directory
from lafla_ai_core.runtime.context import ChatMessage
from lafla_ai_core.runtime.generation_contract import (
    build_generation_request,
    decode_completion_from_ids,
    trim_completion_token_ids,
)
from lafla_ai_core.runtime.output_guard import OutputGuardResult
from lafla_ai_core.runtime.policy import render_runtime_output


DEFAULT_SYSTEM_TEXT = (
    "Sen Lafla dil modeli ailesinden bir yardımcı modelsin. "
    "Verilen model kimliğini koru, sınırlarını açıkça belirt ve doğrulanmamış kesin iddia üretme."
)
BLOCKING_CHECKPOINT_WARNINGS = (
    "low_information_completion",
    "empty_after_output_guard",
    "possible_prompt_leak",
    "possible_mojibake",
    "safety_filters_disabled",
)


@dataclass(frozen=True)
class CheckpointQualityAssessment:
    """Checkpoint smoke sonucunun release adayi olup olmadigini tasir."""

    ok: bool
    blocking_warnings: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class CheckpointGenerationResult:
    """Checkpoint smoke generation sonucunu tasir."""

    checkpoint_dir: str
    prompt_text: str
    public_text: str
    warnings: tuple[str, ...]
    generated_token_count: int
    device: str
    quality_ok: bool
    blocking_warnings: tuple[str, ...]
    raw_thinking: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


class TokenizersGenerationAdapter:
    """Hugging Face tokenizers.Tokenizer nesnesini generation sozlesmesine uyarlar."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]:
        try:
            encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)  # type: ignore[attr-defined]
        except TypeError:
            encoded = self._tokenizer.encode(text)  # type: ignore[attr-defined]
        if hasattr(encoded, "ids"):
            return tuple(int(token_id) for token_id in encoded.ids)
        return tuple(int(token_id) for token_id in encoded)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens))  # type: ignore[attr-defined]


def build_checkpoint_messages(user_text: str, system_text: str = DEFAULT_SYSTEM_TEXT) -> tuple[ChatMessage, ...]:
    """Checkpoint smoke testi icin Lafla sohbet mesajlarini uretir."""

    user = user_text.strip()
    system = system_text.strip()
    if not user:
        raise ValueError("prompt bos olamaz")
    if not system:
        raise ValueError("system_text bos olamaz")
    return (ChatMessage("system", system), ChatMessage("user", user))


def build_model_system_text(model_config: ModelConfig) -> str:
    """Checkpoint model kimligini typed config alanlarindan uretir."""

    identity = model_config.identity_statement or model_config.display_name or model_config.name
    return (
        f"{identity} "
        "Model kimliğini tutarlı koru, sınırlarını açıkça belirt ve doğrulanmamış kesin iddia üretme."
    )


def assess_checkpoint_generation_quality(public_text: str, warnings: Sequence[str]) -> CheckpointQualityAssessment:
    """Checkpoint smoke generation sonucunu fail-closed kalite kararina cevirir."""

    blocking: list[str] = []
    for warning in warnings:
        if warning in BLOCKING_CHECKPOINT_WARNINGS and warning not in blocking:
            blocking.append(str(warning))
    if not public_text.strip() and "empty_after_output_guard" not in blocking:
        blocking.append("empty_public_text")
    if blocking:
        return CheckpointQualityAssessment(False, tuple(blocking), f"blocking_warnings:{','.join(blocking)}")
    return CheckpointQualityAssessment(True, (), "ok")


def generate_from_checkpoint(
    *,
    checkpoint_dir: str | Path,
    tokenizer_path: str | Path,
    user_text: str,
    system_text: str | None = None,
    max_new_tokens: int = 64,
    device: str | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> CheckpointGenerationResult:
    """Checkpoint'ten greedy smoke generation yapar ve public output guard sonucunu dondurur."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens pozitif olmali")
    if runtime_config is not None:
        runtime_config.validate()
    checkpoint = Path(checkpoint_dir)
    tokenizer_file = Path(tokenizer_path)
    validate_checkpoint_directory(checkpoint)
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"tokenizer bulunamadi: {tokenizer_file}")

    try:
        import torch
        from tokenizers import Tokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("checkpoint generation icin torch ve tokenizers paketleri gerekli") from exc

    from lafla_ai_core.model.transformer import LaflaDecoderModel

    model_config = ModelConfig.from_mapping(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    model_config.validate()
    resolved_system_text = system_text.strip() if system_text is not None else build_model_system_text(model_config)
    if not resolved_system_text:
        raise ValueError("system_text bos olamaz")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LaflaDecoderModel(model_config).to(resolved_device)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=resolved_device))
    model.eval()

    tokenizer = TokenizersGenerationAdapter(Tokenizer.from_file(str(tokenizer_file)))
    request = build_generation_request(build_checkpoint_messages(user_text, resolved_system_text), tokenizer)
    generated_ids = list(request.prompt_token_ids)
    active_stop_token_ids = () if runtime_config is not None and runtime_config.safety_profile == "off" else request.stop_token_ids
    stopped = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=resolved_device)
            logits = model(input_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_id)
            if active_stop_token_ids and _ends_with_stop(generated_ids, active_stop_token_ids):
                stopped = True
                break
    guarded, raw_thinking = _render_checkpoint_completion(
        tokenizer,
        generated_ids,
        prompt_token_count=len(request.prompt_token_ids),
        stop_token_ids=active_stop_token_ids,
        stopped=stopped,
        prompt_text=user_text,
        system_text=resolved_system_text,
        runtime_config=runtime_config,
    )
    quality = assess_checkpoint_generation_quality(guarded.text, guarded.warnings)
    return CheckpointGenerationResult(
        checkpoint_dir=str(checkpoint),
        prompt_text=user_text,
        public_text=guarded.text,
        warnings=guarded.warnings,
        generated_token_count=max(0, len(generated_ids) - len(request.prompt_token_ids)),
        device=str(resolved_device),
        quality_ok=quality.ok,
        blocking_warnings=quality.blocking_warnings,
        raw_thinking=raw_thinking,
    )


def _render_checkpoint_completion(
    tokenizer: TokenizersGenerationAdapter,
    generated_ids: Sequence[int],
    *,
    prompt_token_count: int,
    stop_token_ids: Sequence[Sequence[int]],
    stopped: bool,
    prompt_text: str,
    system_text: str,
    runtime_config: RuntimeConfig | None,
) -> tuple[OutputGuardResult, str | None]:
    if runtime_config is None:
        guarded = decode_completion_from_ids(
            tokenizer,
            generated_ids,
            prompt_token_count=prompt_token_count,
            stop_token_ids=stop_token_ids,
            prompt_text=prompt_text,
            system_text=system_text,
        )
        return guarded, None

    if runtime_config.safety_profile == "off":
        completion_ids = tuple(int(token_id) for token_id in generated_ids[prompt_token_count:])
    else:
        completion_ids = trim_completion_token_ids(
            generated_ids,
            prompt_token_count=prompt_token_count,
            stop_token_ids=stop_token_ids,
        )
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    output = render_runtime_output(raw_text, runtime_config, prompt_text=prompt_text, system_text=system_text)
    warnings = output.warnings
    if stopped and "control_token_stop" not in warnings:
        warnings = (*warnings, "control_token_stop")
    return OutputGuardResult(text=output.public_text, warnings=warnings), output.raw_thinking


def _ends_with_stop(token_ids: Sequence[int], stop_token_ids: Sequence[Sequence[int]]) -> bool:
    values = tuple(int(token_id) for token_id in token_ids)
    for stop in stop_token_ids:
        stop_tuple = tuple(int(token_id) for token_id in stop)
        if stop_tuple and len(values) >= len(stop_tuple) and values[-len(stop_tuple) :] == stop_tuple:
            return True
    return False
