# Dizin Disiplini

## Ust Kural

Her klasor bir sorumluluk siniri tasir. Klasor birikme yeri degildir.

## Kaynak Agaci

```text
src/lafla_ai_core/
  cli/             komut girisleri
  config/          config semasi ve dogrulama
  data/            manifest, lisans, PII, shard
  tokenizer/       tokenizer egitimi ve kalite kapilari
  model/           model cekirdegi
  training/        pretraining dongusu ve checkpoint
  post_training/   SFT, DPO, preference
  evaluation/      kalite ve guvenlik kapilari
  runtime/         generation ve sohbet
  export/          quantization ve paketleme
  colab/           Colab operasyon yardimcilari
  observability/   log, rapor, health
```

## Repo Kategorileri

```text
configs/
  data/
    identity/       model kimligi ve davranis seed kimlikleri
    source-plans/   gercek veri kaynak plani ve oran sozlesmeleri
    *.yaml          data policy gibi kategori ustu sozlesmeler
  training/
    colab/          Colab TPU/T4 profilleri
    kaggle/         Kaggle GPU profilleri
    lightning/      Lightning/H200 profilleri

datasets/
  post_training/
    thinking/       thinking SFT seedleri ve manifestleri
      jsonl/        egitime verilecek JSONL formatli SFT seedleri
      manifests/    uretim ve kullanim sozlesmesi manifestleri
    safety/         jailbreak/safety SFT seedleri ve manifestleri
      jsonl/        safety SFT JSONL seedleri
      manifests/    safety seed manifestleri
  pretraining/      buyuk corpus icin belge/sozlesme; buyuk veri commit edilmez
  evaluation/       eval set belgeleri ve kucuk gate girdileri

scripts/
  colab/            Colab launcher scriptleri
  kaggle/           Kaggle launcher scriptleri
  lightning/        Lightning launcher scriptleri
  data/             gercek veri hazirlama araclari

notebooks/
  colab/            Colab notebooklari

artifacts/
  benchmarks/       RSS/performans benchmark raporlari
  reports/
    validation/     veri ve SFT validation raporlari
```

Yeni dosya eklerken dosyanin "nerede calistigi" ile "ne ise yaradigi" ayrilir.
Ornek: Kaggle launcher `scripts/kaggle/` altina, Kaggle training config
`configs/training/kaggle/` altina gider. Egitimde kullanilan gercek corpus repo
icine koyulmaz; sadece policy, source plan, manifest ornegi veya kucuk identity
seedleri tutulur.

`datasets/post_training/` altindaki hicbir JSONL dosyasi base pretraining
karisimina girmez. Bu dosyalar SFT/kalibrasyon asamasina aittir; pretraining
CLI, runner ve quality scan bu ayrimi fail-closed korur.

## Yasak Isimler

- `misc`
- `temp`
- `new`
- `common2`
- `helpers`
- `stuff`
- `old`

Eski kod gerekiyorsa `docs/references` altinda belgelenir; kaynak agaca kirli
tasima yapilmaz.
