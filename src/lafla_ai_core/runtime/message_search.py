"""
@Dosya: runtime/message_search.py
@Aciklama: Yetki kapsamli mesaj arama ve kayit butunlugunu koruyan token paketleme.
@Uyari: API ham SQL, shell komutu veya kapsamsiz tum-hesap aramasi kabul etmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from lafla_ai_core.runtime.conversation_memory import MESSAGE_FRAME_TOKENS


TokenCounter = Callable[[str], int]
MAX_RETRIEVAL_TOKENS = 2048


@dataclass(frozen=True)
class MessageSearchScope:
    actor_id: str
    platform: str
    conversation_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.actor_id.strip():
            raise ValueError("actor_id bos olamaz")
        if self.platform not in {"instagram", "discord"}:
            raise ValueError("platform instagram veya discord olmali")
        if not self.conversation_ids:
            raise ValueError("en az bir yetkili conversation id zorunlu")


@dataclass(frozen=True)
class MessageSearchQuery:
    text: str
    scope: MessageSearchScope
    limit: int = 20

    def validate(self) -> None:
        self.scope.validate()
        if not self.text.strip():
            raise ValueError("arama metni bos olamaz")
        if not 1 <= self.limit <= 100:
            raise ValueError("arama limiti 1-100 araliginda olmali")


@dataclass(frozen=True)
class SearchHit:
    message_id: str
    conversation_id: str
    content: str
    score: float
    platform: str
    timestamp: str

    def validate(self, scope: MessageSearchScope) -> None:
        if not self.message_id.strip() or not self.conversation_id.strip():
            raise ValueError("search hit kimlikleri bos olamaz")
        if not self.content.strip():
            raise ValueError("search hit content bos olamaz")
        if self.platform != scope.platform:
            raise PermissionError(f"search hit platform kapsam disi: {self.platform}")
        if not self.timestamp.strip():
            raise ValueError("search hit timestamp zorunlu")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("search hit score 0-1 araliginda olmali")


@dataclass(frozen=True)
class PackedSearchResults:
    query: MessageSearchQuery
    hits: tuple[SearchHit, ...]
    token_count: int


class MessageSearchBackend(Protocol):
    def search(self, query: MessageSearchQuery) -> Sequence[SearchHit]:
        """Yalniz query.scope icindeki mesajlari dondurur."""


def pack_search_results(
    query: MessageSearchQuery,
    hits: Sequence[SearchHit],
    *,
    token_counter: TokenCounter,
    max_tokens: int = MAX_RETRIEVAL_TOKENS,
) -> PackedSearchResults:
    query.validate()
    if not 1 <= max_tokens <= MAX_RETRIEVAL_TOKENS:
        raise ValueError(f"retrieval max_tokens 1-{MAX_RETRIEVAL_TOKENS} araliginda olmali")
    allowed_conversations = set(query.scope.conversation_ids)
    ranked = sorted(hits, key=lambda hit: (-hit.score, hit.message_id))[: query.limit]
    packed: list[SearchHit] = []
    token_count = 0
    for hit in ranked:
        hit.validate(query.scope)
        if hit.conversation_id not in allowed_conversations:
            raise PermissionError(f"kapsam disi conversation: {hit.conversation_id}")
        cost = token_counter(hit.content) + MESSAGE_FRAME_TOKENS
        if cost > max_tokens or token_count + cost > max_tokens:
            continue
        packed.append(hit)
        token_count += cost
    return PackedSearchResults(query, tuple(packed), token_count)


def search_messages(
    backend: MessageSearchBackend,
    query: MessageSearchQuery,
    *,
    token_counter: TokenCounter,
    max_tokens: int = MAX_RETRIEVAL_TOKENS,
) -> PackedSearchResults:
    query.validate()
    return pack_search_results(
        query,
        backend.search(query),
        token_counter=token_counter,
        max_tokens=max_tokens,
    )
