"""
@Dosya: export/hf_package.py
@Aciklama: Tokenizer ve runtime metadata icin Hugging Face uyumlu paket uretir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu paket model agirligini tek basina standart Transformers mimarisine donusturmez.
@Calisma-Semasi: tokenizer.json -> HF metadata files
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.export.hf_remote_code import build_hf_config_payload, remote_code_files
from lafla_ai_core.tokenizer.chat_template import build_hf_chat_template


SPECIAL_TOKEN_KEYS = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
ADDITIONAL_SPECIAL_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|think|>", "<|/think|>")


def write_hf_tokenizer_package(
    tokenizer_json_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    model_config: ModelConfig | None = None,
) -> Path:
    """Tokenizer metadata dosyalarini HF model repo klasorune yazar."""

    source = Path(tokenizer_json_path)
    if not source.exists():
        raise FileNotFoundError(f"tokenizer.json bulunamadi: {source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_payload = json.loads(source.read_text(encoding="utf-8"))
    vocab = _extract_vocab(tokenizer_payload)
    _require_tokens(vocab)

    shutil.copy2(source, output / "tokenizer.json")
    _write_json(
        output / "special_tokens_map.json",
        {
            **SPECIAL_TOKEN_KEYS,
            "additional_special_tokens": list(ADDITIONAL_SPECIAL_TOKENS),
        },
    )
    _write_json(
        output / "tokenizer_config.json",
        {
            "model_max_length": model_config.context_length if model_config is not None else 2048,
            "chat_template": build_hf_chat_template(),
            "clean_up_tokenization_spaces": False,
            "add_bos_token": False,
            "add_eos_token": False,
            **SPECIAL_TOKEN_KEYS,
            "additional_special_tokens": list(ADDITIONAL_SPECIAL_TOKENS),
        },
    )
    _write_json(
        output / "generation_config.json",
        {
            "bos_token_id": vocab["<|bos|>"],
            "eos_token_id": vocab["<|eos|>"],
            "pad_token_id": vocab["<|pad|>"],
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "use_cache": True,
            "cache_implementation": "dynamic",
        },
    )
    if model_config is not None:
        _write_json(output / "config.json", build_hf_config_payload(model_config, vocab))
        for file_name, content in remote_code_files().items():
            (output / file_name).write_text(content, encoding="utf-8")
    (output / "README.md").write_text(_model_card(model_name, model_config=model_config), encoding="utf-8")
    return output


def _extract_vocab(tokenizer_payload: dict[str, Any]) -> dict[str, int]:
    model = tokenizer_payload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("tokenizer.json model.vocab mapping tasimali")
    vocab: dict[str, int] = {}
    for token, token_id in model["vocab"].items():
        vocab[str(token)] = int(token_id)
    return vocab


def _require_tokens(vocab: dict[str, int]) -> None:
    missing = [token for token in (*SPECIAL_TOKEN_KEYS.values(), *ADDITIONAL_SPECIAL_TOKENS) if token not in vocab]
    if missing:
        raise ValueError(f"HF paket icin tokenizer special token eksik: {missing}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_card(model_name: str, *, model_config: ModelConfig | None = None) -> str:
    target_line = ""
    display_line = ""
    creator_line = ""
    identity_line = ""
    if model_config is not None:
        target_line = f"- Beyan edilen hedef parametre: {model_config.parameter_target:,}.\n"
        if model_config.display_name:
            display_line = f"- Gorunen model adi: {model_config.display_name}.\n"
        if model_config.creator_name:
            creator_line = f"- Gelistiren: {model_config.creator_name}.\n"
        if model_config.identity_statement:
            identity_line = f"- Kimlik cumlesi: {model_config.identity_statement}\n"
    return (
        f"# {model_name}\n\n"
        "Bu klasor Lafla tokenizer ve runtime metadata paketidir.\n\n"
        "## Model kimligi\n\n"
        f"- Model adi: `{model_name}`.\n"
        f"{display_line}"
        f"{creator_line}"
        f"{identity_line}"
        f"{target_line}\n"
        "## Kullanim\n\n"
        "- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` ve "
        "`generation_config.json` Hugging Face tokenizer yuklemesi icin hazirdir.\n"
        "- Mevcut Lafla decoder mimarisi ozel oldugu icin agirliklar standart "
        "`AutoModelForCausalLM` ile kod donusumu olmadan garanti edilmez.\n"
        "- Modelleme dosyalari HF reposuna eklendiginde kullanici yerel LaflaAi-Core "
        "reposuna ihtiyac duymadan `trust_remote_code=True` ile yukleyebilir.\n"
    )
