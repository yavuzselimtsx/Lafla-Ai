# Datasets

Bu klasor veri dosyalarini amac ve asamaya gore ayirir. Rastgele JSONL veya
manifest dosyasi dogrudan `datasets/` altina konmaz.

Beklenen agac:

```text
datasets/
  pretraining/              buyuk gercek corpus yerlesimi icin belge ve kucuk ornekler
  post_training/
    thinking/
      jsonl/                thinking SFT JSONL seedleri
      manifests/            thinking SFT manifestleri
    safety/
      jsonl/                jailbreak/safety SFT JSONL seedleri
      manifests/            jailbreak/safety SFT manifestleri
  evaluation/               eval set belgeleri ve kucuk eval girdileri
```

Gercek pretraining corpus dosyalari repo icine commit edilmez. Colab/Kaggle
calisma alaninda `LaflaAI100M/data/train.jsonl` ve
`LaflaAI100M/data/veri_manifesti.json` olarak tutulur.
