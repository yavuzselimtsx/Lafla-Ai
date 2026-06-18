# Datasets

Bu klasör veri dosyalarını amaç ve eğitim aşamasına göre ayırır. Rastgele JSONL
veya manifest dosyası doğrudan `datasets/` altına konmaz.

Beklenen ağaç:

```text
datasets/
  pretraining/
    curated/
      jsonl/                küçük, doküman biçimli continued-pretraining seedleri
      manifests/            kaynak hash'i ve kalite ölçümleri
  post_training/
    chat/
      jsonl/                kalite odaklı, dengeli sohbet SFT seedleri
      manifests/            kategori, dil, tekrar ve refusal ölçümleri
    thinking/
      jsonl/                eski thinking SFT seedleri
      manifests/            eski thinking SFT manifestleri
    safety/
      jsonl/                düşük oranlı jailbreak/safety SFT seedleri
      manifests/            jailbreak/safety SFT manifestleri
  evaluation/               eval set belgeleri ve küçük eval girdileri
```

Gerçek büyük pretraining corpus dosyaları repo içine commit edilmez. Çalışma
alanında `LaflaAI100M/data/train.jsonl` ve
`LaflaAI100M/data/veri_manifesti.json` olarak tutulur.

Temel sınır: `post_training/` altındaki hiçbir dosya `train_pretrain
--data-jsonl` listesine eklenmez. `pretraining/curated/` ise yalnız düşük
ağırlıklı continued-pretraining seedidir; ana gerçek corpusun yerine geçmez.
