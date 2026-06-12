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
  mobile/          dusuk guclu cihaz paket hedefleri
  safety/          runtime politika ve cikti kapilari
  observability/   log, rapor, health
```

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

