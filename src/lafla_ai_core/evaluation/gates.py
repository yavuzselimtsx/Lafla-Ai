"""
@Dosya: evaluation/gates.py
@Aciklama: LaflaAi-Core release icin zorunlu kalite kapilarini tanimlar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Loss dusmesi tek basina model kalitesi degildir; release gate raporu
        tokenizer, Turkce kimlik, guvenlik ve dusuk guc calistirmayi birlikte
        degerlendirir.
@Uyari: Gate basarisizsa checkpoint urun adayi sayilamaz.
@Calisma-Semasi: gate results -> aggregate -> pass/fail
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class GateResult:
    """Tek kalite kapisi sonucunu tasir."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    """Release gate toplu raporunu tasir."""

    ok: bool
    pass_rate: float
    results: tuple[GateResult, ...]

    def to_json(self) -> str:
        """Raporu JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def aggregate_gates(results: Iterable[GateResult], minimum_pass_rate: float = 1.0) -> GateReport:
    """Gate sonuc listesini release kararina cevirir."""

    result_tuple = tuple(results)
    if not result_tuple:
        return GateReport(ok=False, pass_rate=0.0, results=())
    passed = sum(1 for result in result_tuple if result.passed)
    pass_rate = passed / len(result_tuple)
    return GateReport(ok=pass_rate >= minimum_pass_rate, pass_rate=pass_rate, results=result_tuple)


def required_gate_results(required_names: Iterable[str], observed: Iterable[GateResult]) -> GateReport:
    """Eksik gate'leri otomatik fail ederek rapor uretir."""

    observed_by_name = {result.name: result for result in observed}
    expanded: list[GateResult] = []
    for name in required_names:
        expanded.append(observed_by_name.get(name, GateResult(name=name, passed=False, detail="gate sonucu yok")))
    return aggregate_gates(expanded, minimum_pass_rate=1.0)

