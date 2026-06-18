# Post-Training Datasets

Bu alan pretraining corpus değildir. SFT, davranış ve safety seedleri burada
kategorize edilir.

```text
chat/jsonl/          yeni kalite odaklı sohbet SFT seedleri
chat/manifests/      kategori, dil, tekrar ve refusal kalite raporları
thinking/jsonl/      eski thinking ve belirsizlik davranışı seedleri
thinking/manifests/  eski thinking seed manifestleri
safety/jsonl/        jailbreak ve policy dirençlilik seedleri
safety/manifests/    safety seed manifestleri
```

Yeni önerilen başlangıç kaynağı `chat/` hattıdır. Üretici dağılımı cevaplanabilir
anchor `%46`, format takibi `%12`, Türkçe `%8`, Almanca `%8`, bounded
uncertainty `%6`, kimlik `%3`, bot bağlamı `%6`, kod kalitesi `%6` ve safety
`%5` olarak sınırlar. Manifest; tam kopya, user/assistant benzersizliği,
cevaplanabilir soruda refusal, dil sızıntısı ve mojibake ölçümlerini taşır.

20K büyük artifact üretimi:

```powershell
$env:PYTHONPATH='src'; python -m lafla_ai_core.cli.generate_quality_chat_seed --count 20000 --output artifacts/generated_datasets/lafla-mini-quality-chat-seed-20k.jsonl --manifest artifacts/generated_datasets/lafla-mini-quality-chat-seed-20k.manifest.json
```

SFT karışımı hazırlanırken safety oranı `%10`, uncertainty oranı `%8` ve toplam
refusal oranı `%16` sınırlarını aşamaz. Cevaplanabilir bir örnekte refusal veya
Türkçe/Almanca dil sızıntısı tek kayıt olsa bile kaliteyi düşürür.

Launcher'lar bu dosyaları doğrular, ancak `train_pretrain --data-jsonl`
listesine eklemez. Eski `thinking/` ve `safety/` dosyaları yeni karışıma körlemesine
eklenmemeli; önce `sft_mixture` kalite raporundan geçmelidir.
