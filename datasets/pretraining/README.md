# Pretraining Datasets

Büyük gerçek pretraining corpus dosyaları bu repoya commit edilmez. Bu klasör
yerleşim sözleşmesini ve küçük, denetlenebilir continued-pretraining seedlerini
taşır.

Çalışma ortamlarında beklenen ana dosyalar:

```text
LaflaAI100M/data/train.jsonl
LaflaAI100M/data/veri_manifesti.json
```

Kaynak ve oran sözleşmeleri `configs/data/source-plans/` altındadır.

`curated/` hattı `Prompt.md` içindeki gerçek kaynak bloklarından tekli, ikili ve
üçlü dokümanlar oluşturur. Sohbet rolleri veya üretilmiş yeni olgu eklemez;
kaynak hash'i, kapsama, tekrar ve benzersizlik ölçümlerini manifeste yazar.

Önerilen 1K yerel artifact üretimi:

```powershell
$env:PYTHONPATH='src'; python -m lafla_ai_core.cli.generate_curated_pretraining_seed --source ..\..\Prompt.md --count 1000 --output artifacts/generated_datasets/lafla-prompt-curated-pretraining-seed-1k.jsonl --manifest artifacts/generated_datasets/lafla-prompt-curated-pretraining-seed-1k.manifest.json
```

Tek bir kaynak belgeden 10K satır üretmek teknik olarak mümkün olsa da kaynak
bloklarını gereksiz sık tekrar ettirir. Bu nedenle doğrulanmış 1K çıktı düşük
ağırlıklı seed olarak kullanılır; genel dil ve dünya bilgisi için lisansı ve
kaynağı denetlenmiş gerçek corpus gerekir.
