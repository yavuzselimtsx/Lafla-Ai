# LaflaAi-Core Mimari Omurgasi

## Ilke

LaflaAi-Core tek bir egitim betigi degildir. Her katmanin sorumlulugu ayridir ve
Colab yalnizca bu katmanlari cagirir.

```text
manifest -> data audit -> tokenizer -> pretraining -> sft -> dpo -> eval
   |            |              |            |          |      |      |
   v            v              v            v          v      v      v
source      reports       tokenizer      ckpt       sft    pref   gates
catalog     + shards      package        state      model  model  release
```

## Katmanlar

`config`
- Model, tokenizer, egitim, export ve runtime ayarlarini dogrular.
- Magic number egitim koduna gomulmez.

`data`
- Manifest okur, kaynak lisansi dogrular, PII/duplikasyon/kalite filtresi uygular.
- Her shard icin kaynak raporu ve token butcesi uretir.

`tokenizer`
- Turkce karakter, kod blogu, URL, sistem/sohbet/thinking tokenlari icin kalite
  kapilari tasir.
- Tokenizer raporu gecmeden model egitimi baslamaz.

`model`
- Decoder-only transformer cekirdegidir.
- RoPE, RMSNorm, SwiGLU, GQA, QK norm gibi secimler configten gelir.
- Persona veya guvenlik politikasi model sinifina gomulmez.

`training`
- Pretraining dongusu, checkpoint, resume, optimizer, gradient accumulation,
  mixed precision ve health log sorumlulugunu tasir.
- Colab, Windows ve Linux kosulari ayni config sozlesmesini kullanir.

`post_training`
- SFT, DPO ve ileride tercih/odul recetelerini ayri tutar.
- Thinking SFT kayitlarini JSONL sozlesmesiyle dogrular.
- Sistem/kullanici tokenlari loss disinda kalir; assistant ve opsiyonel private
  thinking segmentleri label-mask ile yonetilir.
- Public runtime cevabi private `<|think|>` bloklarini sizdirmaz.

`evaluation`
- Loss disinda Turkce kalite, kimlik tutarliligi, halusinasyon, kod kalitesi,
  guvenlik ve uzun baglam kapilarini kosar.

`runtime`
- Chat, generation, sampling ve conversation state yonetir.
- DEV modu ham dusunceyi gosterebilir; PROD modu politika katmanindan gecer.
- Thinking effort, thinking budget, developer mode ve context overflow stratejisi
  configten gelir.
- Sistem promptu context penceresinde korunur; context dolunca eski non-system
  mesajlar kontrollu dusurulur.

`export`
- Safetensors, GGUF benzeri CPU dostu ciktilar, quantization ve artifact raporunu
  yonetir.

`colab`
- Drive mount, checkpoint arsivleme, resume, kota ve runtime guard mantigini
  modullerden cagirir. Defter icine kalici is mantigi gomulmez.

`mobile`
- Dusuk guclu CPU ve mobil hedeflerde model paketi, bellek butcesi ve quantized
  runtime kararlarini tutar.

## Referans Kaynak Kullanimi

OLMo Core ve GPT-NeoX temiz oda referansidir.

Alinacak dersler:
- OLMo Core: konfigurasyon disiplini, moduler train/eval/optim ayrimi, script
  aileleri.
- GPT-NeoX: buyuk model egitim konfigurasyonu, distributed kavramlari, checkpoint
  araclari ve post-training ayrimi.

Alinmayacak seyler:
- Kod kopyasi.
- Dosya agacinin aynen tasinmasi.
- Lisans metni veya test fixture tasima.
- Lafla Prompt kurallarini delen monolitik egitim akisi.
