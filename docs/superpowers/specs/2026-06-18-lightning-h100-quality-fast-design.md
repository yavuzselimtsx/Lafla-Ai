# Lightning H100 Kalite Korumalı Hız Profili Tasarımı

Tarih: 2026-06-18
Durum: Kullanıcı tarafından onaylandı

## Amaç

LaflaGPT Mini 100M pretraining sürecini boş bir Lightning Studio üzerindeki tek NVIDIA H100 80GB HBM3 cihazında, mevcut öğrenme dinamiklerini değiştirmeden mümkün olan en yüksek güvenli verimle sürdürmek.

Eğitim, GitHub Release içindeki doğrulanmış `lafla-step-019500` saf pretraining checkpoint'inden devam edecek. SFT checkpoint'leri korunacak fakat pretraining kaynağı olarak kullanılmayacak.

## Başarı Ölçütleri

- H100 üzerinde BF16, fused AdamW ve native GQA etkin olmalı.
- Hedef token sayısı, öğrenme oranı çizelgesi ve optimizer başına 32 dizi korunmalı.
- Her curriculum aşamasında per-device micro-batch yeniden çözülmeli.
- Restore edilen checkpoint model, optimizer, RNG, trainer state ve `READY.json` içermeli.
- Tokenizer, veri ve checkpoint SHA-256 doğrulaması geçmeden eğitim başlamamalı.
- Eğitim terminalden bağımsız arka planda çalışmalı; PID, GPU, sağlık logu, checkpoint ve backup durumu tek komutla izlenebilmeli.
- Her normal, final ve kesinti checkpoint'i ayrı snapshot olarak korunmalı.
- OOM geri çekilmesi hiçbir kısmi optimizer güncellemesini sürdürmemeli; süreç son sağlam checkpoint'ten daha küçük batch profiliyle tekrar başlamalı.

## Kapsam Dışı

- SFT veya safety eğitimi başlatmak.
- Pretraining corpusunu ya da tokenizer sözleşmesini değiştirmek.
- Öğrenme oranını batch büyüklüğüne göre ölçeklemek.
- SFT ağırlıklarını pretraining checkpoint'iyle birleştirmek.
- Çoklu GPU/DDP davranışını yeniden tasarlamak.

## Seçilen Yaklaşım

Sabit micro-batch yerine curriculum aşamasına bağlı deterministik H100 geometrisi kullanılacak:

| Sequence length | Per-device micro-batch | Gradient accumulation | Optimizer başına dizi |
| ---: | ---: | ---: | ---: |
| 2048 | 32 | 1 | 32 |
| 4096 | 16 | 2 | 32 |
| 8192 | 8 | 4 | 32 |
| 12288 | 4 | 8 | 32 |
| 16384 | 2 | 16 | 32 |
| 20480 | 1 | 32 | 32 |

Bu seçim kısa context aşamasında H100'ı beslerken uzun context aşamalarında activation belleğini sınırlar. Global optimizer batch ve dolayısıyla eğitim semantiği mevcut kalite profilindeki değerle aynı kalır.

## Yapılandırma Sözleşmesi

`TrainingConfig` aşağıdaki isteğe bağlı alanı destekleyecek:

```yaml
cuda_micro_batch_size_per_device_curriculum:
  - 32
  - 16
  - 8
  - 4
  - 2
  - 1
```

Kurallar:

- Liste boşsa mevcut sabit `cuda_micro_batch_size_per_device` davranışı korunur.
- Liste doluysa uzunluğu `sequence_curriculum` ile aynı olmalıdır.
- Her değer pozitif olmalı ve aktif CUDA cihazlarıyla oluşan global micro-batch, `target_sequences_per_optimizer_step` değerini tam bölmelidir.
- Curriculum dışı eğitimde bu alan kullanılamaz.
- H100 profilinde `precision: bf16`, `prefer_fused_optimizer: true` ve `target_sequences_per_optimizer_step: 32` kullanılacak.

## Eğitim Döngüsü

Runner başlangıçta resume checkpoint'indeki `cumulative_tokens` değerinden aktif curriculum aşamasını bulacak ve batch geometrisini o aşama için çözecek.

Curriculum geçişinde şu işlemler atomik sıra ile yapılacak:

1. Yeni sequence length belirlenir.
2. Yeni per-device micro-batch ve gradient accumulation çözülür.
3. Gradient checkpointing kararı güncellenir.
4. Veri iteratorü yeni sequence length ve micro-batch ile yeniden kurulur.
5. `curriculum_transition` sağlık kaydı yeni geometriyi içerir.

Optimizer, scheduler ve model ağırlıkları yeniden oluşturulmaz. Böylece curriculum geçişi yalnız veri/batch geometrisini değiştirir.

## OOM Geri Çekilmesi

H100 launcher, ana eğitimi şu sıralı batch ölçekleriyle çalıştırabilir: `1.0`, `0.5`, `0.25`. Bir ölçek tüm curriculum micro-batch değerlerine uygulanır; sonuçlar en az 1 olacak şekilde 32'nin bölenlerine indirilir ve gradient accumulation yeniden hesaplanır.

Geri çekilme yalnız CUDA OOM açıkça saptanırsa yapılır. Diğer hatalarda süreç fail-closed durur. OOM sonrası:

- Çalışan süreç tamamen kapanır ve CUDA belleği serbest bırakılır.
- Son `READY.json` içeren checkpoint tekrar seçilir.
- Daha küçük ölçekle yeni süreç başlatılır.
- Seçilen ölçek log ve sağlık kayıtlarına yazılır.

Kısmi adım veya bellekteki optimizer durumu kullanılmaz.

## Bootstrap ve Restore Akışı

Yeni Lightning Studio için H100 bootstrap script'i:

1. H100 cihaz adını, compute capability değerini, BF16 desteğini, sürücüyü ve boş diski doğrular.
2. Repo yoksa `main` dalını klonlar; varsa fast-forward günceller. Kirli veya ayrışmış repo durumunda durur.
3. Ayrı H100 virtualenv oluşturur ve resmî CUDA 12.8 wheel kanalından varsayılan olarak `torch==2.11.0` kurar. Sürüm yalnız açık bir `LAFLA_TORCH_VERSION` değişkeniyle değiştirilebilir ve kurulum sonrasında H100 compute capability `9.0` ile BF16 desteği yeniden doğrulanır.
4. GitHub Release `lafla-100m-backup-20260618` içinden yalnız ana eğitim için gereken dosyaları indirir:
   - `pretrain-lafla-step-019500.tar.zst`
   - `runtime-metadata.tar.zst`
   - `SHA256SUMS`
5. İndirilen dosyaları manifest ile doğrular ve geçici dizinde açar.
6. Checkpoint sözleşmesini, tokenizer vocab boyutunu, veri manifestini ve gerçek veri kalite kapılarını doğrular.
7. Doğrulanan dosyaları `LaflaAI100M` çalışma köküne atomik olarak taşır.
8. Eğitimi arka planda başlatır.

Eksik artifact, checksum uyuşmazlığı, SFT checkpoint yolu, tokenizer uyumsuzluğu veya gerçek veri doğrulama hatası sessiz fallback üretmez. Resume kaynağının `trainer_state.json` içindeki `format` değeri tam olarak `lafla-trainer-state-v2` olmalı; `lafla-thinking-sft-state-v1` ve bilinmeyen formatlar reddedilmelidir.

## Launcher ve İzleme

H100 için üç giriş noktası bulunacak:

- Ön planda kurulum ve eğitim launcher'ı.
- `nohup` kullanan arka plan launcher'ı.
- H100 log adlarını da otomatik bulan ortak monitor.

Launcher aynı anda ikinci pretraining sürecini başlatmayacak. PID dosyası yalnız süreç oluşturulduktan sonra yazılacak. Nohup logu, health JSONL, aktif config, resume checkpoint ve batch ölçeği raporlanacak.

## Checkpoint Dayanıklılığı

- Token tabanlı checkpoint aralığı `10,000,000` olacak.
- Son sekiz aktif checkpoint retention altında tutulacak.
- Retention öncesinde ve her kayıt sonrasında ayrı `checkpoint-backups` snapshot'ı oluşturulacak.
- Kaynak checkpoint veya mevcut backup üzerine yazılmayacak.
- Disk alt sınırı H100 çalışma alanına uygun biçimde en az 20GB olacak.
- Kesinti sinyali geldiğinde son tamamlanan adım ayrı interrupted checkpoint olarak saklanacak.

## Test Tasarımı

### Birim testleri

- Curriculum micro-batch listesinin parse ve validation kuralları.
- Altı aşamanın beklenen `32/1` ile `1/32` geometrilerini üretmesi.
- Resume sırasında doğru aşamanın ve geometrinin seçilmesi.
- Curriculum geçişinde iterator ve batch geometrisinin birlikte yenilenmesi.
- Geçersiz bölünebilirlik, uzunluk ve sıfır değerlerinin reddedilmesi.
- BF16 CUDA yolunda fused AdamW seçimi.

### Launcher testleri

- H100 profilinin kalite sabitlerini T4/RTX profilleriyle karşılaştırma.
- SHA-256 doğrulaması başarısızsa extraction ve eğitim yapılmaması.
- SFT klasörünün resume kaynağı olarak reddedilmesi.
- CUDA OOM dışındaki hatalarda otomatik retry yapılmaması.
- Arka plan launcher'ın çift süreç başlatmaması.
- Monitor'un H100 nohup logunu göstermesi.

### Uzak smoke doğrulaması

- H100, BF16, fused optimizer ve native GQA sağlık kaydında görünmeli.
- Resume olayı `lafla-step-019500` ve doğru cumulative token değerini göstermeli.
- İlk gerçek eğitim kayıtlarında loss sonlu olmalı ve optimizer başına dizi 32 kalmalı.
- GPU kullanımı, VRAM, checkpoint ve snapshot dizinleri uzaktan doğrulanmalı.

## Geri Alma

Yeni H100 profili ve launcher bağımsız dosyalardır. Schema alanı isteğe bağlı olduğu için eski T4, RTX Pro 6000, Kaggle ve TPU profilleri değişmeden çalışmaya devam eder. Sorun halinde H100 süreci durdurulur; kaynak `019500` checkpoint'i ve GitHub Release değişmeden kalır.
