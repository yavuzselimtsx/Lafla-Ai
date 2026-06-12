# LaflaGPT Mini 100M Düşük Güç Runtime

## Hedef Cihaz

- Intel i3-1215U
- 8 GB sistem RAM
- CPU inference
- Tek eşzamanlı generation

## Bellek Tasarımı

20.480 token context için config’ten türetilen çekirdek tahmin:

- INT8 ağırlık: 98.324.736 bayt
- BF16 hibrit GQA KV cache: 50.331.648 bayt
- Python/PyTorch/Transformers ve allocator headroom ayrıca bütçelenir

Bu değerler tahmindir. Kabul ölçütü worker, tokenizer, CLI ve aynı süreçteki tüm
çocukları içeren gerçek peak RSS değerinin 700 MiB’yi aşmamasıdır. Harici ayrı
veritabanı servisi dahil değildir; aynı CLI içindeki indeks istemcisi dahildir.

## Context Yönetimi

- Model kapasitesi: 20.480 token
- Özetleme tetikleyicisi: 15.360 token
- Korunan son mesajlar: en fazla 4.096 token
- Structured summary: en fazla 2.048 token
- Instagram/Discord mesaj retrieval: en fazla 2.048 token
- Cevap rezervi: 2.048 token

Özet doğrulanmadan eski mesajlar aktif context’ten çıkarılmaz. Tam transcript
mesaj deposunda kalır.

## Benchmark

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.benchmark_inference `
  --model-dir C:\LaflaAI100M\hf-package `
  --profiles 2048 8192 15360 20000 `
  --output artifacts\benchmarks\lafla-100m-rss.json
```

Her profil ayrı süreçte prefill ve bir decode adımı çalıştırır. Ana süreç ve
çocukların peak RSS/USS değerleri raporlanır. Model/dependency yoksa sonuç
`blocked`; sınır aşılırsa `failed` olur.

Normal 2K-4K bot konuşmasında 8 token/s hedefi yalnız gerçek cihaz ölçümüyle
release kapısı sayılır.
