# Colab Stratejisi

## Hedef

Colab defteri egitim mantiginin kendisi degil, tekrar uretilebilir bir calistirma
yuzeyidir. Her kritik davranis repo modullerinde bulunur.

## Zorunlu Akis

1. Gercek Drive mount kontrolu yap.
2. Repo/zip hash ve klasor kokunu dogrula.
3. Manifest, tokenizer ve config preflight calistir.
4. Checkpoint dizinini Drive veya bilerek secilmis local fallback olarak isaretle.
5. Her checkpointten sonra state, log ve manifest raporu yaz.
6. Runtime kapanmadan once final artifact arşivi ve `EGITIM_NOTU.txt` uret.

## Mount Kuralı

`/content/drive` doluysa oraya mount denenmez. Temiz mount noktasi kullanilir:

```python
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
```

Gercek Drive yolu:

```text
/content/gdrive/MyDrive/LaflaAI/...
```

## Checkpoint Kuralı

Checkpoint basarili sayilmak icin sunlari tasir:

- `model.safetensors`
- `config.json`
- tokenizer paketi veya tokenizer path raporu
- `training_log.jsonl`
- `train_state`
- `checkpoint_report.json`
- veri manifest hashleri

## Kota Kuralı

Colab T4 hedefleri icin varsayilanlar:

- 400M sinifi model.
- FP16/bfloat16 uygunluk kontrolu.
- 8-bit optimizer opsiyonel.
- Gradient accumulation ile bellek dengeleme.
- Checkpoint retention varsayilan 3.
