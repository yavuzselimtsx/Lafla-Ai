# LaflaGPT Mini 100M Eğitim Planı

## 1. Zorunlu Girdiler

- `configs/data/source-plans/lafla-100m-source-plan.json` aday kaynak ve hedef oran planıdır.
- Gerçek shard manifesti URL, snapshot, lisans, hash, filtre raporu ve token
  sayısını taşımalıdır.
- `review_required` kaynaklar insan incelemesi tamamlanmadan indirilemez veya
  primary karışıma alınamaz.
- Sahte/bootstrap veri oluşturulmaz.

Hedef başlangıç karışımı:

| Alan | Oran |
| --- | ---: |
| Türkçe yüksek kaliteli genel/eğitsel | %42 |
| Almanca yüksek kaliteli genel/eğitsel | %23 |
| İngilizce yüksek kaliteli genel/eğitsel | %10 |
| Türkçe/Almanca ansiklopedik-tarihsel | %10 |
| Matematik, mantık ve bilim | %8 |
| Kod ve teknik dokümantasyon | %7 |

## 2. Preflight

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.quality_scan --root .
PYTHONPATH=src python -m lafla_ai_core.cli.preflight \
  configs/model/lafla-100m-thinking.yaml \
  configs/training/colab/colab-tpu-v5e-100m.yaml \
  configs/tokenizer/turkish-german-thinking-bpe.yaml \
  configs/runtime/desktop-i3-int8-100m.yaml \
  configs/post_training/lafla-thinking-sft.yaml
PYTHONPATH=src python -m lafla_ai_core.cli.training_phase_plan
```

## 3. Tokenizer

- 32.768 byte-level BPE
- UTF-8 NFC
- Türkçe ve Almanca roundtrip/fertility raporu
- Chat, thinking, tool ve document sınır tokenları
- Mojibake, URL ve kod blokları kalite kapıları

## 4. TPU Pretraining

Colab çalışma alanında gerçek `train.jsonl` ve `veri_manifesti.json` hazır
olduktan sonra:

```bash
cd /content/LaflaAi-Core
bash scripts/colab/start_tpu_v5e_100m.sh
```

Token curriculum:

| Başlangıç tokenı | Sequence |
| ---: | ---: |
| 0 | 2.048 |
| 3,6B | 4.096 |
| 4,8B | 8.192 |
| 5,4B | 12.288 |
| 5,7B | 16.384 |
| 5,9B | 20.480 |

İlk hedef 6B gerçek tokendir. Uzun vadeli checkpoint devam hedefi 20B+ tokendir.
Tek Colab oturumu tamamlanmış eğitim kabul edilmez.

## 5. Uncertainty ve Instruction SFT

- Cevaplanabilir/cevaplanamaz eşleşmiş örnekler
- Çelişkili retrieval kanıtları
- Kaynaksız güncel bilgi soruları
- Yanlış öncülü düzeltme
- Türkçe ve Almanca doğal diyalog
- Kaynakta olmayan ayrıntı üretmeme
- Private thinking ile public cevabın ayrılması

Mevcut yerel SFT çekirdekleri:

- `configs/data/identity/lafla-model-identity-100m.jsonl`: küçük model kimliği ve
  belirsizlik davranışı sohbet çekirdeği.
- `configs/post_training/lafla-100m-seed-profile.json`: sentetik seedlerin tek
  model/dil/davranış profil kaynağı. 200M veya yeni dil denemelerinde kod
  değiştirilmez; ayrı profil dosyası verilir.
- `datasets/post_training/thinking/jsonl/lafla-100m-thinking-chat-seed-20k.jsonl`: 20.000 satırlık
  sentetik sohbet/thinking seed. Manifesti `allowed_for_pretraining: false`
  taşır; gerçek corpus yerine değil, kısa SFT kalibrasyonu için kullanılır.
- `datasets/post_training/safety/jsonl/lafla-100m-safety-jailbreak-seed-10k.jsonl`: 10.000
  satırlık güvenlik, jailbreak direnci, sistem prompt sızdırma reddi, özel veri
  sınırı, tool izni ve halüsinasyon reddi seed'i. Manifesti yine
  `allowed_for_pretraining: false` taşır.

Doğrulama:

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.validate_thinking_sft \
  --input datasets/post_training/thinking/jsonl/lafla-100m-thinking-chat-seed-20k.jsonl \
  --report artifacts/reports/validation/lafla-100m-thinking-chat-seed-20k.validation.json \
  --max-thinking-chars 900
PYTHONPATH=src python -m lafla_ai_core.cli.validate_thinking_sft \
  --input datasets/post_training/safety/jsonl/lafla-100m-safety-jailbreak-seed-10k.jsonl \
  --report artifacts/reports/validation/lafla-100m-safety-jailbreak-seed-10k.validation.json \
  --max-thinking-chars 1200
```

Colab launcher bu iki dosyayı post-training girdisi olarak doğrular ve rapora
yazar. Bunlar base pretraining `--data-jsonl` karışımına sokulmaz; SFT aşaması
assistant loss maskesiyle çalışmalıdır.

Ek koruma: `train_pretrain` CLI ve runner, `datasets/post_training/` altindan
gelen girdiyi fail-closed reddeder. `quality_scan` da script/notebook/src icinde
`--data-jsonl datasets/post_training/` kalibini hata sayar.

## 6. Release

Loss düşmesi tek başına yeterli değildir. Release için Almanca/Türkçe kalite,
abstention, kaynak sadakati, 20K needle/passkey, cache eşitliği, role/prompt
sızıntısı, jailbreak direnci, sistem prompt sızdırma reddi, güvenli tool sınırı
ve kullanıcının i3-1215U cihazında gerçek process-tree peak RSS kapıları geçilmelidir.
