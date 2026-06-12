"""
@Dosya: data/routing.py
@Aciklama: Dataset dosyalarinin hangi egitim asamasinda kullanilabilecegini
            fail-closed kurallarla dogrular.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Sequence


POST_TRAINING_PREFIX = "datasets/post_training/"
IDENTITY_PREFIX = "configs/data/identity/"


def assert_pretraining_inputs(paths: Sequence[str]) -> None:
    """Pretraining'e yanlislikla post-training/thinking/safety verisi sokulmasini engeller."""

    for value in paths:
        normalized = _normalize_path(value)
        if normalized.startswith(POST_TRAINING_PREFIX):
            raise ValueError(
                "post-training dataset pretraining corpus'una karistirilamaz: "
                f"{value}. Bu dosya SFT/kalibrasyon asamasinda kullanilmali."
            )


def _normalize_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")
