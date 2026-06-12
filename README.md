# LaflaAi-Core

LaflaAi-Core, Türkçe ve Almanca odaklı LaflaGPT Mini 100M teknik profilinin temiz oda eğitim,
değerlendirme, Transformers export ve düşük maliyetli inference çekirdeğidir.
Eski daha büyük profiller karşılaştırma ve legacy çalışma amacıyla korunur; aktif
Colab yolu 100M profildir.

## LaflaGPT Mini 100M Mimari

- 98.324.736 tahmini parametre
- 12 katman, 768 hidden, 2.048 SwiGLU
- 12 query head, 2 KV head
- 9 local katman (`sliding_window=4096`), 3 global katman
- 32.768 vocabulary
- 20.480 token model context
- Transformers remote code, açık KV cache ve INT8 masaüstü profili

Model GPT-5 sınıfı kapasite iddiasında bulunmaz. Hedef, 100M sınıfında güçlü
Türkçe/Almanca, ölçülü mantık, kaynak sadakati ve doğru belirsizlik davranışıdır.

## Runtime Sözleşmesi

- Toplam context: 20.480 token
- Otomatik özetleme eşiği: 15.360 token
- Korunan son konuşma: en fazla 4.096 token, mesajlar bölünmez
- Mesaj arama/RAG: inference çağrısı başına en fazla 2.048 token
- Cevap rezervi: 2.048 token
- Aynı worker içinde eşzamanlı generation: 1
- Kabul sınırı: ana süreç ve tüm çocukların ölçülen peak RSS değeri en fazla 700 MiB

Bellek formülü yalnız planlama içindir. Release kararı gerçek
`process-tree peak RSS` benchmark sonucuyla verilir.

## Yerel Doğrulama

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.quality_scan --root .
python -m lafla_ai_core.cli.preflight `
  configs/model/lafla-100m-thinking.yaml `
  configs/training/colab/colab-tpu-v5e-100m.yaml `
  configs/tokenizer/turkish-german-thinking-bpe.yaml `
  configs/runtime/desktop-i3-int8-100m.yaml `
  configs/post_training/lafla-thinking-sft.yaml
python -m lafla_ai_core.cli.training_phase_plan
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Sentetik SFT Seedleri

Küçük identity dosyası `configs/data/identity/lafla-model-identity-100m.jsonl` altında
vardır. Ek sentetik SFT seedleri `configs/post_training/lafla-100m-seed-profile.json`
profilinden tekrar üretilebilir; model kimliği, dil odağı ve davranış metinleri
Python koduna gömülmez.

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.generate_synthetic_chat_seed `
  --profile configs\post_training\lafla-100m-seed-profile.json
python -m lafla_ai_core.cli.generate_safety_jailbreak_seed `
  --profile configs\post_training\lafla-100m-seed-profile.json
python -m lafla_ai_core.cli.validate_thinking_sft `
  --input datasets\post_training\thinking\jsonl\lafla-100m-thinking-chat-seed-20k.jsonl `
  --report artifacts\reports\validation\lafla-100m-thinking-chat-seed-20k.validation.json `
  --max-thinking-chars 900
python -m lafla_ai_core.cli.validate_thinking_sft `
  --input datasets\post_training\safety\jsonl\lafla-100m-safety-jailbreak-seed-10k.jsonl `
  --report artifacts\reports\validation\lafla-100m-safety-jailbreak-seed-10k.validation.json `
  --max-thinking-chars 1200
```

Bu dosyalar gerçek pretraining corpus'u değildir. Manifestleri
`allowed_for_pretraining: false` taşır; kullanım yeri kısa post-training /
davranış ve güvenlik kalibrasyonudur. Colab 100M launcher bu seedleri doğrular ve
`reports/post-training-sft-inputs.json` içine kaydeder; `train_pretrain`
komutunun `--data-jsonl` listesine sentetik SFT seed eklenmez.

## Colab TPU v5e

Gerçek eğitimden önce şu dosyalar zorunludur:

```text
/content/LaflaAI100M/data/train.jsonl
/content/LaflaAI100M/data/veri_manifesti.json
```

Launcher sahte/bootstrap veri üretmez ve eksik dosyada kapanır:

```bash
cd /content/LaflaAi-Core
bash scripts/colab/start_tpu_v5e_100m.sh
```

Notebook:

```text
notebooks/colab/lafla_tpu_100m_training.ipynb
```

Resume:

```bash
export RESUME_FROM=/content/LaflaAI100M/checkpoints/lafla-step-010000
bash scripts/colab/start_tpu_v5e_100m.sh
```

Curriculum toplam görülen token sayısına göre 2K, 4K, 8K, 12K, 16K ve 20K
aşamalarına geçer. Trainer state, optimizer/RNG yanında `cumulative_tokens`,
`curriculum_stage` ve aktif `sequence_length` alanlarını taşır.

## Hugging Face Export

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.hf_package `
  --tokenizer-json C:\LaflaAI100M\tokenizer\lafla-tokenizer.json `
  --output-dir C:\LaflaAI100M\hf-package `
  --model-config configs/model/lafla-100m-thinking.yaml `
  --model-name lafla-100m-thinking
```

Paket `trust_remote_code=True` ile çalışan config/model dosyalarını, local/global
attention metadata’sını ve cache-aware generation kodunu içerir.

## Gerçek RSS Benchmark

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.benchmark_inference `
  --model-dir C:\LaflaAI100M\hf-package `
  --output artifacts\benchmarks\lafla-100m-rss.json
```

Model klasörü, Torch, Transformers veya psutil eksikse rapor `blocked` ve
`accepted:false` olur; tahmini değer başarı sayılmaz.

## 100M → 200M

200M hedef sıfırdan başlamaz. Uyumlu 100M checkpointinin 12 katmanı korunur,
17 yeni residual blok identity-compatible eklenir:

```powershell
$env:PYTHONPATH='src'
python -m lafla_ai_core.cli.grow_checkpoint `
  --source-checkpoint C:\LaflaAI100M\checkpoints\lafla-final `
  --target-model-config configs/model/lafla-200m-thinking.yaml `
  --output-checkpoint C:\LaflaAI200M\checkpoints\growth-init
```

Bu çıktı yalnız başlangıç checkpointidir; 200M model için devam pretraining
zorunludur.
