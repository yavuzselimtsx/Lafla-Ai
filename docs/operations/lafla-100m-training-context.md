# LaflaGPT Mini 100M Training Context

Tarih: 2026-06-18

Bu dosya context sıfırlanınca LaflaGPT Mini 100M çalışmasının neden bu şekilde
ilerlediğini hızlı anlamak için yazıldı. Buradaki notlar operasyonel hafızadır;
kanıt gerektiren durumlarda remote makine, GitHub release ve repo dosyaları
yeniden doğrulanmalıdır.

## Model Hedefi

- Model adı kullanıcı tarafında `LaflaGPT Mini` olarak seçildi. `LaflaGPT 100M`
  ürün adı olarak kullanılmamalı.
- Mimari hedef küçük 100M sınıfı modeldir; amaç GPT-5.5 seviyesinde gerçek
  kapasite iddia etmek değil, 100M sınıfında mümkün olan en iyi Türkçe/Almanca
  pretraining + daha sonra SFT/prompt-following kalibrasyonu yapmaktır.
- Modelin kimlik bilgisi: LaflaGPT Mini, Yavuz Selim tarafından geliştirilen
  küçük Lafla modeli; GPT-5.5 değildir.
- Kullanıcının ana kalite istekleri: daha az halüsinasyon, bilmiyorsa uydurmama,
  Türkçe ve Almanca iyi davranış, bağlam takibi, mantık yürütme, Discord ve
  Instagram botu gibi CLI/API runtime senaryolarına uygunluk.
- Kod içine model davranışı için hardcoded cevap yazılmamalı. Davranış veriden,
  tokenizer/model checkpoint sözleşmesinden ve post-training aşamasından gelmeli.

## Değiştirilmemesi Gereken Sözleşmeler

- Pretraining checkpoint üstünde tokenizer değiştirmek güvenli değildir. Tokenizer
  değişirse embedding sözleşmesi bozulur. Tokenizer değişimi ancak yeni bir
  training ailesi sıfırdan başlatılırken yapılmalıdır.
- SFT checkpoint pretraining resume kaynağı değildir. Pretraining resume için
  `trainer_state.json` formatı `lafla-trainer-state-v2` olmalı; SFT formatı
  `lafla-thinking-sft-state-v1` olan dizinler base training'e sokulmamalıdır.
- `datasets/post_training/` altındaki thinking/safety/chat verileri base
  pretraining `--data-jsonl` karışımına sokulmamalı. Bunlar SFT içindir.
- Sahte/bootstrap veri ile eğitim başlatılmamalı. Veri yoksa hazırlama açıkça
  gerçek kaynaklardan yapılmalı veya fail-closed durmalıdır.

## Veri Durumu

H100 oturumunda GitHub release içinde `train.jsonl` asset'i olmadığı için veri
remote makinede yeniden hazırlandı. Hazırlama raporu başarılıydı:

- `dataset_version`: `lafla-100m-lightning-h100-realdata-2026-06`
- `total_chars`: `166015433`
- `total_records`: `61413`
- `unique_hashes`: `61413`
- Kaynaklar: Türkçe FineWeb2/HQ, Almanca FineWeb2/HQ, Türkçe Wikimedia,
  Almanca tarih Wikimedia, OpenWebMath.
- `fineweb2_hq_english` builder config hatasıyla alınamadı.
- `the_stack_smol_python` gated dataset olduğu için opsiyonel olarak atlandı.

Veri hazırlama bittikten sonra Hugging Face dataset finalization sırasında Python
abort verdi, fakat `train.jsonl`, `veri_manifesti.json` ve report `ok=true`
üretilmişti. Launcher tekrar başlatıldı ve mevcut veri dosyalarını kullanarak
pretraining'e geçti.

## Pretraining Zaman Çizgisi

### Eski T4/Lightning Akışı

- T4 üzerinde uzun süre tek GPU ile pretraining yapıldı.
- Eski hız profilinde `effective_micro_batch_size=2`,
  `gradient_accumulation_steps=16`, `sequences_per_optimizer_step=32` idi.
- Erken checkpoint testleri kötüydü; raw pretraining checkpointlerinden chat
  kalitesi beklenmemeli.

### H100 Kalite-Hız Profili

Repo commit:

```text
327cb1c9ce8f6212f2e902f527c24409e4fbfb6f
```

H100 için eklenen profil:

```text
configs/training/lightning/lightning-h100-100m-quality-fast.yaml
scripts/lightning/start_h100_100m.sh
scripts/lightning/start_h100_100m_background.sh
scripts/lightning/monitor_h100_100m.sh
scripts/lightning/bootstrap_h100_100m.sh
```

H100 pretraining, GitHub release'den restore edilen saf pretraining checkpoint
`lafla-step-019500` üzerinden başlatıldı. İlk başarılı H100 health kayıtlarında:

- `cuda_batch_scale`: `1.0`
- `effective_micro_batch_size`: `32`
- `effective_gradient_accumulation_steps`: `1`
- `sequences_per_optimizer_step`: `32`
- `optimizer_mode`: `fused_adamw`
- `native_gqa`: `true`
- `sequence_length`: `2048`

Bu profil kaliteyi korumak için optimizer başına 32 sequence sözleşmesini
korur; H100 hızını micro-batch'i artırarak kullanır.

## Checkpoint ve Yedekler

Ana GitHub release:

```text
https://github.com/yavuzselimtsx/Lafla-Ai/releases/tag/lafla-100m-backup-20260618
```

Önemli assetler:

- `pretrain-lafla-step-019500.tar.zst`: H100 başlangıç pretraining checkpoint'i.
- `runtime-metadata.tar.zst`: tokenizer ve eski runtime metadata.
- `h100-lafla-step-027771-emergency-20260618-164339.tar.zst`: tek checkpoint
  acil yedeği.
- `h100-all-checkpoints-lafla-step-028534-emergency-20260618-165658.tar.zst.part-000`
  ile `part-004`: tüm H100 checkpoint seti parçalı yedeği.
- `SHA256SUMS`, `RESTORE.txt`, `BACKUP_MANIFEST.json`: restore ve doğrulama
  bilgileri.

Parçalı full yedekte 8 adet READY checkpoint doğrulandı:

```text
lafla-step-027771
lafla-step-027924
lafla-step-028000
lafla-step-028077
lafla-step-028229
lafla-step-028382
lafla-step-028500
lafla-step-028534
```

Son full yedek anındaki en son READY checkpoint:

```text
lafla-step-028534
```

Restore komutu:

```bash
cat h100-all-checkpoints-lafla-step-028534-emergency-20260618-165658.tar.zst.part-* \
  | zstd -dc \
  | tar -C /teamspace/studios/this_studio -xf -
```

## SFT Durumu

SFT denemeleri 19.500 pretraining checkpoint'i üstünden ayrı output dizinlerine
yapıldı. Bunlar base pretraining yerine geçmez.

Release'deki SFT assetleri:

```text
sft-lafla-step-019500-thinking-sft-balanced-v3.tar.zst
sft-lafla-step-019500-chat-sft-balanced-v4.tar.zst
sft-lafla-step-019500-chat-anchor-clean-v1.tar.zst
sft-lafla-step-019500-chat-anchor-clean-v1-corrected.tar.zst
sft-lafla-step-019500-chat-anchor-clean-v1-corrected-unknown.tar.zst
```

Test için en güncel tercih edilen SFT asset:

```text
sft-lafla-step-019500-chat-anchor-clean-v1-corrected-unknown.tar.zst
```

Bu SFT modelleri erken checkpoint üstündeki kaliteyi görmek için yapılmış ara
denemelerdir. Final chat modeli olarak kabul edilmemelidir. Erken SFT'lerde bazı
kalıplara kilitlenme, Ankara/kimlik gibi örnekleri ezberlemiş gibi davranma ve
bilmediğini söyleme davranışında zayıflık görüldü.

## Test Beklentileri

- `lafla-step-028534` ham pretraining checkpointidir. Chat/prompt following
  beklenmemeli; soru-cevapta saçmalaması normaldir.
- Ham checkpoint testinde `quality_ok=false`, `smoke_answer_drift` veya anlamsız
  metin görürsen bu tek başına training bozuk demek değildir. SFT görmemiş base
  modeldir.
- SFT checkpoint testi için scriptlerde `set -e` kullanılmamalı; test CLI
  semantic veya structural failure döndürünce sonraki promptlar çalışmadan script
  kesilebilir. Kullanıcıya test bloğu verirken `set +e` kullan.
- CPU testleri yavaş olabilir. H100 hakkı bittiğinde CPU test sadece smoke amaçlıdır.

Örnek raw checkpoint test:

```bash
cd /teamspace/studios/this_studio/LaflaAi-Core
source /teamspace/studios/this_studio/.venvs/lafla-100m-h100/bin/activate 2>/dev/null || true
export PYTHONPATH=src

python -m lafla_ai_core.cli.test_checkpoint \
  --checkpoint-dir /teamspace/studios/this_studio/LaflaAI100M/checkpoints/lafla-step-028534 \
  --tokenizer-path /teamspace/studios/this_studio/LaflaAI100M/tokenizer/lafla-tokenizer.json \
  --prompt "Türkiye'nin başkenti neresidir? Sadece şehir adını yaz." \
  --max-new-tokens 96 \
  --temperature 0.7 \
  --top-k 40 \
  --repetition-penalty 1.12 \
  --seed 42 \
  --device cpu
```

Örnek SFT test hazırlığı:

```bash
SFT_ASSET=sft-lafla-step-019500-chat-anchor-clean-v1-corrected-unknown.tar.zst
SFT_URL="https://github.com/yavuzselimtsx/Lafla-Ai/releases/download/lafla-100m-backup-20260618/$SFT_ASSET"
SFT_BASE=/teamspace/studios/this_studio/LaflaAI100M/sft-test
mkdir -p "$SFT_BASE/extracted-corrected-unknown"
curl --fail --location --retry 5 --retry-all-errors -o "$SFT_BASE/$SFT_ASSET" "$SFT_URL"
zstd -t "$SFT_BASE/$SFT_ASSET"
tar --use-compress-program=zstd -xf "$SFT_BASE/$SFT_ASSET" -C "$SFT_BASE/extracted-corrected-unknown"
find "$SFT_BASE/extracted-corrected-unknown" -type f -name model.pt -printf '%h\n' | head -n 1
```

## Neden Bazı Şeyler Yapılmadı?

- 100M checkpoint üstünde tokenizer değiştirilmedi: embedding sözleşmesi bozulurdu.
- SFT ağırlıkları pretraining'e karıştırılmadı: format ve eğitim amacı farklıdır.
- Safety verisi base pretraining'e sokulmadı: küçük modelde refusal kalıbını baskın
  hale getirip genel konuşmayı bozabilir.
- Model davranışı için kod içine sabit cevap eklenmedi: ileride 200M veya yeni dil
  ailesi eğitilirken eski davranışın koda gömülü kalması kurumsal ve teknik borçtur.
- H100 profilinde LR batch'e göre büyütülmedi: kaliteyi korumak için optimizer
  başına sequence sayısı sabit tutuldu.

## Sonraki Mantıklı Adımlar

1. `lafla-step-028534` ham checkpointi kısa test et, ama chat kalitesi hükmünü SFT
   olmadan verme.
2. En güncel SFT asset'i CPU smoke ile test et.
3. Bir sonraki GPU oturumunda `lafla-step-028534` veya daha yeni full yedekten
   resume et.
4. Yeni SFT yapılacaksa safety oranı düşük, çeşitli, shuffle edilmiş ve prompt
   following ağırlıklı olmalı. Tek şablon tekrarından kaçın.
5. Final kalite için ayrı evaluation set gerekir: kimlik, matematik, Ankara,
   Almanca, bilmiyorum/abstention, jailbreak/prompt injection, uzun bağlam ve RAG.
