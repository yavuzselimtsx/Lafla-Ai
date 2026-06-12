"""
@Dosya: tokenizer/chat_template.py
@Aciklama: Lafla sohbet role token sozlesmesini tek yerde toplar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Egitim, SFT ve Hugging Face export ayni sohbet render kurallarini kullanir.
@Uyari: Kullanici metni role veya control token tasiyamaz.
@Calisma-Semasi: turns -> Lafla chat text / HF chat template
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


THINK_OPEN = "<|think|>"
THINK_CLOSE = "<|/think|>"
ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
CONTROL_TOKENS = ("<|bos|>", "<|eos|>", THINK_OPEN, THINK_CLOSE, *ROLE_TOKENS.values())


@dataclass(frozen=True)
class ChatTurn:
    """Tek sohbet turunu tasir."""

    role: str
    content: str


def render_chat_transcript(
    turns: Iterable[ChatTurn],
    *,
    include_bos: bool = True,
    include_eos: bool = True,
    validate_content_tokens: bool = True,
) -> str:
    """Sohbet turlarini Lafla special-token metnine cevirir."""

    parts: list[str] = []
    if include_bos:
        parts.append("<|bos|>")
    turn_count = 0
    for turn in turns:
        turn_count += 1
        if turn.role not in ROLE_TOKENS:
            raise ValueError(f"desteklenmeyen chat role: {turn.role}")
        content = turn.content.strip()
        if not content:
            raise ValueError(f"bos chat icerigi: {turn.role}")
        if validate_content_tokens:
            validate_no_control_tokens(content, turn.role)
        parts.append(f"{ROLE_TOKENS[turn.role]}\n{content}")
    if turn_count == 0:
        raise ValueError("en az bir chat turn gerekli")
    if include_eos:
        parts.append("<|eos|>")
    return "\n".join(parts)


def render_generation_prompt(turns: Iterable[ChatTurn]) -> str:
    """Inference icin assistant cevabini baslatan prompt uretir."""

    transcript = render_chat_transcript(turns, include_bos=True, include_eos=False)
    return f"{transcript}\n<|assistant|>\n"


def validate_no_control_tokens(text: str, context: str) -> None:
    """Kullanici/sistem iceriginde control token bulunmasini engeller."""

    for token in CONTROL_TOKENS:
        if token in text:
            raise ValueError(f"{context}: control token kullanilamaz: {token}")


def build_hf_chat_template() -> str:
    """Transformers `tokenizer_config.json` icin Jinja chat template dondurur."""

    return (
        "{% if bos_token is defined and bos_token %}{{ bos_token }}{% else %}<|bos|>{% endif %}\n"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] | trim }}\n"
        "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] | trim }}\n"
        "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] | trim }}\n"
        "{% else %}{{ raise_exception('Unsupported Lafla role: ' + message['role']) }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% else %}"
        "{% if eos_token is defined and eos_token %}{{ eos_token }}{% else %}<|eos|>{% endif %}{% endif %}"
    )
