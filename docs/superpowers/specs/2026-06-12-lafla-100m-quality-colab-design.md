# Lafla 100M Kalite, Uzun Bağlam ve Colab TPU Tasarımı

## 1. Amaç

LaflaAi-Core, sıfırdan eğitilecek yaklaşık 100M parametreli, Türkçe ve Almanca
öncelikli bir decoder-only model üretecektir. Model:

- Google Colab TPU üzerinde kesilebilir ve checkpoint'ten devam ettirilebilir.
- Hugging Face Transformers ile `AutoModelForCausalLM` üzerinden çalışır.
- Discord ve Instagram botlarına bağlanan özel bir CLI tarafından kullanılır.
- i3-1215U ve 8 GB RAM sınıfı bir bilgisayarda düşük maliyetle çalışır.
- 20.480 token model bağlamı taşır; normal bot akışında aktif bağlam daha küçük
  tutulur.
- Bilmediği veya yeterli kanıt bulamadığı sorularda tahmin yürütmek yerine
  belirsizliğini açıkça belirtmeyi öğrenir.
- Gelecekte yaklaşık 200M parametreli modele katman ekleme yoluyla bilgi
  aktarılmasına uygun bir model ailesinin ilk üyesidir.

Bu model GPT-5.5 kapasitesine sahipmiş gibi sunulmayacaktır. Hedef, 100M sınıfında
veri kalitesi, Türkçe/Almanca yeterliliği, bağlam yönetimi, ölçülü akıl yürütme ve
ürün güvenilirliği bakımından mümkün olan en güçlü sonucu üretmektir.

## 2. Başarı Ölçütleri

### 2.1 Model

- Gerçek parametre sayısı: 95M-105M.
- Model bağlam kapasitesi: 20.480 token.
- Tokenizer vocabulary: 32.768.
- Türkçe ve Almanca karakter/token roundtrip testleri hatasız.
- Public cevapta sistem, kullanıcı, rol veya private thinking tokenı sızıntısı yok.
- Checkpoint yalnız loss değerine göre değil, dil, muhakeme, belirsizlik, kimlik,
  bağlam ve halüsinasyon kapılarına göre değerlendirilir.

### 2.2 Yerel inference

Hedef makine:

- Intel Core i3-1215U, 6 çekirdek/8 izlek, AVX2.
- 8 GB sistem RAM.
- Windows ve SSD.

Ana dağıtım profili:

- Transformers tabanlı CPU inference.
- INT8 weight-only quantization.
- Batch size 1.
- KV cache açık.
- Tek bot yanıtı için tek aktif generation.

Kabul kapıları:

- CLI, Transformers, PyTorch, tokenizer, KV cache, retrieval payload ve generation
  çalışma alanını içeren inference işlem ağacının peak RSS değeri 700 MiB'yi
  aşmaz.
- USS ve bileşen tahminleri ayrıca raporlanır; kabul kararı peak RSS ile verilir.
- 2K, 8K, 15K ve 20K prompt profilleri OOM olmadan tamamlanır.
- Normal 2K-4K bot konuşmasında hedef en az 8 token/s'dir. Bu bir tahmin değil,
  kullanıcının i3-1215U cihazında ölçülecek release kapısıdır.
- 15K özetleme eşiğinde bot kilitlenmez; özetleme süresi ve ilk-token gecikmesi
  benchmark raporuna yazılır.

Harici bir veritabanı ayrı servis olarak çalışıyorsa onun belleği model worker
RSS hesabına katılmaz. Aynı CLI işlemi içinde çalışan mesaj deposu veya indeks
istemcisi hesaba katılır.

## 3. Model Mimarisi

Önerilen başlangıç yapılandırması:

| Alan | Değer |
| --- | --- |
| Aile | Decoder-only causal LM |
| Katman | 12 |
| Hidden size | 768 |
| Intermediate size | 2048 |
| Attention heads | 12 |
| KV heads | 2 |
| Head dimension | 64 |
| Vocabulary | 32.768 |
| Context | 20.480 |
| Aktivasyon | SwiGLU |
| Norm | RMSNorm |
| Konum | RoPE + uzun bağlam ölçekleme |
| Embedding | Input/output tied |
| Bias | Kapalı |
| Dropout | Config ile, pretraining varsayılanı 0 |

Bu yapı yaklaşık 98.3M parametredir. Parametre hesabı config preflight sırasında
yeniden yapılır; model adı parametre sayısının kanıtı sayılmaz.

### 3.1 Hibrit attention

- 12 katmanın 9'u 4.096 token sliding-window attention kullanır.
- Her dört katmandan biri global attention kullanır: 3 global katman.
- GQA, 12 query head ve 2 KV head ile KV cache maliyetini düşürür.
- Attention düzeni config'te katman deseni olarak tanımlanır; model koduna sabit
  liste gömülmez.
- Eğitim ve HF export aynı attention desenini kullanır.

Bu seçim OLMo Core'daki katman bazlı sliding-window/full-attention düzeni ve GQA
ayrımından temiz oda tasarım dersi alır. Kod kopyalanmaz.

### 3.2 Transformers cache sözleşmesi

Mevcut HF remote-code çıktısındaki `use_cache=false` kaldırılacaktır. Model:

- `past_key_values` kabul eder.
- Yalnız işlenmemiş yeni tokenları decode adımında işler.
- Transformers `Cache` sözleşmesiyle Dynamic ve Static cache destekler.
- INT8 ağırlık profiliyle BF16/FP16 veya desteklenen quantized KV cache
  seçeneklerini benchmark eder.
- Cache açık ve kapalı generation çıktılarının aynı olduğunu test eder.

KV cache olmadan uzun yanıt üretmek kabul edilmez; her yeni tokenda tüm promptu
yeniden hesaplayan yol release dışıdır.

### 3.3 Tahmini çekirdek bellek

20.480 context ve batch size 1 için yaklaşık değerler:

- FP16 ağırlıklar: 188 MiB.
- INT8 ağırlıklar: 94 MiB.
- Hibrit BF16 KV cache: yaklaşık 48 MiB.
- INT8 ağırlık + BF16 KV çekirdeği: yaklaşık 142 MiB.

700 MiB bütçenin kalanı Python, PyTorch, Transformers, tokenizer, temporary
tensors, allocator parçalanması ve CLI bileşenleri içindir. Tahmini değer release
kanıtı değildir; gerçek peak RSS testi zorunludur.

## 4. Bağlam ve Konuşma Belleği

### 4.1 Token bütçesi

Toplam model bağlamı 20.480 tokendır. Bir CLI çağrısındaki bütçe:

1. Sistem ve model kimliği.
2. Önceki yapılandırılmış konuşma özeti.
3. Mesaj araması/RAG sonucu, en fazla 2.048 token.
4. En yeni konuşma mesajları.
5. Üretilecek cevap için ayrılmış çıktı bütçesi.

Bu bölümlerin tamamı aynı 20.480 sınırından harcar. Retrieval 2.048 tokenı aşarsa
en yüksek skorlu parçalar korunur; metin sessizce ortadan kesilmez.

### 4.2 Otomatik özetleme

- Tetikleyici: oluşturulacak prompt 15.360 tokenı aşarsa.
- Sistem mesajı ve son 4.096 token korunur.
- Eski konuşma hiyerarşik olarak özetlenir.
- Yapılandırılmış özet en fazla 2.048 token olur.
- Özet; kullanıcı tercihleri, doğrulanmış gerçekler, kararlar, açık görevler,
  belirsizlikler ve güvenlik açısından gerekli bağlamı ayrı alanlarda taşır.
- Her özet maddesi kaynak mesaj kimliklerini taşır.
- Tam geçmiş Discord/Instagram mesaj deposunda kalır. Yalnız aktif model
  context'inden çıkarılır.
- Özetleme başarısızsa kaynak mesajlar silinmez ve eski özet değiştirilmez.
- Yeni özet doğrulanmadan transaction tamamlanmaz.

Olağan aktif bağlamın özetleme sonrası 6K-10K aralığına dönmesi hedeflenir. 20K
kapasite güvenlik payı ve büyük tekil istekler içindir; her bot mesajında 20K
prefill yapılmaz.

### 4.3 Mesaj araması

CLI'nin Instagram/Discord adaptörü gerektiğinde geçmiş mesajlarda arama yapar.

- Arama isteği modelden gelen serbest shell/SQL metniyle çalışmaz.
- CLI, typed bir `search_messages` araç sözleşmesi sunar.
- Sonuçlarda kanal/kullanıcı yetkisi ve tenant sınırı uygulanır.
- Tek inference çağrısına en fazla 2.048 retrieval tokenı eklenir.
- Sonuçlar mesaj kimliği, zaman, kaynak platform ve güven skoru taşır.
- Model, retrieval sonucu yoksa bunu açıkça görür; olmayan mesaj uydurmaz.

API anahtarları, kullanıcı tokenları ve webhook secret'ları config dosyasına veya
modele gömülmez; environment/secret store üzerinden alınır.

## 5. Bilgi ve Halüsinasyon Stratejisi

### 5.1 Ağırlıklarda tutulacak bilgi

- Güçlü Türkçe ve Almanca dil temeli.
- Türkiye ve Almanya'nın kalıcı coğrafya, tarih, kültür ve kurum bilgileri.
- Dünya tarihi; Adolf Hitler ve İkinci Dünya Savaşı gibi konular dahil, güvenilir
  ansiklopedik ve akademik kaynaklarla dengeli biçimde.
- Temel matematik, mantık, bilim, yazılım ve günlük problem çözme.
- Lafla model kimliği ve geliştirici bilgisi, küçük ve kontrollü SFT kümesiyle.

Kimlik verisi pretraining karışımında aşırı tekrarlanmaz. Kimlik ezberi dil ve
genel bilgi kapasitesini bozmayacak ayrı post-training/eval sözleşmesiyle öğretilir.

### 5.2 RAG'de tutulacak bilgi

2026 haberleri, değişebilen kamu görevlileri, mevzuat, fiyatlar, platform
politikaları ve güncel Türkiye/Almanya verileri model ağırlıklarının güvenilir
hafızası sayılmaz. Bunlar:

- Güvenilir ve izinli kaynak indeksinden alınır.
- Kaynak zamanı ve URL/kimlik bilgisiyle modele verilir.
- Kaynak bulunamazsa model kesin güncel iddia üretmez.
- Discord/Instagram bot cevabı gerektiğinde kaynak gösterebilir.

### 5.3 “Bilmiyorum” davranışı

Model yalnız “bilmiyorum” cümlesini ezberlemeyecektir. Eğitimde:

- Cevaplanabilir ve cevaplanamaz eşlenmiş örnekler.
- Eksik bağlam örnekleri.
- Çelişkili retrieval örnekleri.
- Güncel bilgi gerektiren ama kaynak verilmeyen örnekler.
- Yanlış öncüllü kullanıcı soruları.
- Güvenli belirsizlik ve açıklayıcı takip sorusu örnekleri

kullanılır.

Release metrikleri:

- Desteksiz kesin iddia oranı.
- Doğru cevaplanabilir soruda gereksiz çekinme oranı.
- Cevaplanamaz soruda doğru abstention oranı.
- Retrieval kaynağına sadakat.
- Kaynakta olmayan ayrıntı ekleme oranı.

## 6. Veri Tasarımı

### 6.1 Pretraining karışımı

Karışım token bütçesiyle ve manifest üzerinden yönetilir. Başlangıç hedef oranları:

| Alan | Hedef |
| --- | ---: |
| Türkçe yüksek kaliteli genel/eğitsel metin | %42 |
| Almanca yüksek kaliteli genel/eğitsel metin | %23 |
| İngilizce yüksek kaliteli genel/eğitsel metin | %10 |
| Türkçe/Almanca ansiklopedik-tarihsel içerik | %10 |
| Matematik, mantık ve bilim | %8 |
| Kod ve teknik dokümantasyon | %7 |

Kaynak adayları:

- FineWeb2 ve FineWeb2-HQ Türkçe/Almanca alt kümeleri.
- Türkçe ve Almanca Wikimedia/Wikipedia dump'ları.
- Lisansı doğrulanmış matematik ve eğitim veri kümeleri.
- İzin/lisans filtresinden geçirilmiş kod veri kümeleri.
- Lafla'ya ait, insan tarafından gözden geçirilmiş küçük veri kümeleri.

`review_required` kaynak otomatik indirilmez. Lisans, kaynak URL'si, snapshot
tarihi, dil, hash, filtre raporu ve token sayısı manifestte olmadan shard eğitime
girmez.

### 6.2 Veri kalite kapıları

- UTF-8 NFC ve mojibake reddi.
- Dil/script doğrulaması.
- Exact ve near-duplicate temizliği.
- PII ve secret taraması.
- Çok kısa, tekrar eden, SEO/spam ve düşük bilgi yoğunluklu metin reddi.
- Eval contamination taraması.
- Türkçe ve Almanca karakter/kelime parçalama raporu.
- Kaynak bazlı maksimum tekrar ve örnek bütçesi.
- Aşırı şiddet/nefret propagandası içeriğinde tarihsel bağlam ve kaynak dengesi.

Sahte bootstrap veri oluşturulmaz. Sentetik/verifier destekli reasoning verisi
kullanılırsa teacher, üretim yöntemi, kaynak problem, doğrulayıcı ve tarih
manifestte zorunludur.

### 6.3 Token bütçesi

Tek bir Colab oturumunda “tamamlandı” varsayımı yapılmaz:

- İlk kullanılabilir temel: en az 2B temiz token.
- Kalite hedefi: 8B-12B benzersiz/iyi karıştırılmış token.
- Uzun vadeli checkpoint devam hedefi: 20B+ token.

Eğitim ilerlemesi yalnız step ile değil, görülen gerçek token, kaynak dağılımı ve
validation eğrileriyle raporlanır.

## 7. Eğitim Aşamaları

### 7.1 Faz A: tokenizer

- 32.768 byte-level BPE.
- Türkçe ve Almanca dengeli örnekleme.
- Chat, rol, tool, think ve doküman sınır tokenları.
- Roundtrip, fertility, unknown, URL, kod ve eklemeli dil testleri.

### 7.2 Faz B: kısa bağlam pretraining

- 2K ardından 4K sequence.
- BF16 TPU/XLA.
- Resume edilebilir optimizer, scheduler, RNG ve veri cursor state.
- Token tabanlı warmup ve cosine decay.
- Her checkpoint Drive'a atomik olarak yazılır.

### 7.3 Faz C: orta ve uzun bağlam

- 8K, 12K, 16K ve 20K aşamalı curriculum.
- Uzun doküman, çok turlu konuşma ve bilgi bulma örnekleri.
- RoPE/uzun bağlam ayarı yalnız kısa smoke ile değil needle, retrieval ve
  passkey eval ile doğrulanır.
- TPU mikro batch ve accumulation değerleri cihaz topolojisi/preflight sonucundan
  türetilir; script içine Colab donanımı varsayımı gömülmez.

### 7.4 Faz D: instruction ve uncertainty SFT

- Türkçe/Almanca doğal konuşma.
- Araç/RAG kullanımı.
- Kimlik tutarlılığı.
- Cevaplanamaz ve çelişkili sorularda abstention.
- Kısa ve doğrulanabilir reasoning/rationale örnekleri.
- Private thinking public cevaptan ayrılır.

100M modelde uzun, gösterişli chain-of-thought üretimi hedeflenmez. Amaç kısa,
kontrollü ve doğrulanabilir mantık adımlarıdır.

### 7.5 Faz E: preference ve release

- İnsan veya güvenilir verifier ile chosen/rejected çiftleri.
- Yanlış kesin cevap, kaynak uydurma, prompt echo, rol sızıntısı ve gereksiz
  tekrar negatif örnektir.
- Release yalnız tüm zorunlu eval kapıları geçerse üretilir.

## 8. 100M'den 200M'ye Büyütme

100M model bağımsız ürün olarak korunur. 200M model:

- Aynı tokenizer, hidden size, head dimension, KV head düzeni ve embedding
  sözleşmesini kullanır.
- Width büyütmek yerine öncelikle katman sayısı artırılarak oluşturulur.
- Yeni katmanlar function-preserving/depth up-scaling dönüşümüyle başlatılır.
- Dönüşüm öncesi ve hemen sonrası logits farkı, perplexity ve regression eval
  raporlanır.
- Ardından düşük öğrenme oranlı continued pretraining uygulanır.

Bu işlem 100M checkpoint'i fiziksel olarak 200M yapmaz; 100M bilgisini başlangıç
olarak kullanan yeni bir 200M checkpoint üretir. 200M için sabit RAM limiti yoktur,
fakat aynı bileşen bazlı bellek tahmini ve gerçek RSS benchmark disiplini sürer.

## 9. Config ve Hardcoded Değer Politikası

Şunlar typed config'ten gelir:

- Model boyutları ve attention deseni.
- Context/summary/retrieval token bütçeleri.
- Cache türü ve dtype.
- Quantization profili.
- Veri kaynakları, ağırlıkları ve token bütçeleri.
- Training curriculum ve checkpoint sıklığı.
- Eval eşikleri.
- CLI platform adaptörleri ve feature flag'leri.

Kod içinde yalnız protokol sabitleri ve güvenlik üst sınırları bulunabilir.
Model adı, 380M/400M yolları, Colab klasörleri, artifact adı, kimlik promptu ve
eğitim step'i üretim koduna gömülmez.

## 10. Hata Yönetimi

- Eksik gerçek veri veya manifest: eğitim başlamaz.
- Lisans/PII/data quality hatası: ilgili shard reddedilir ve raporlanır.
- TPU/XLA kurulumu başarısız: otomatik olarak CPU eğitime düşmez.
- Checkpoint eksik/bozuk: resume reddedilir.
- Özetleme başarısız: mesajlar ve eski özet korunur.
- Retrieval başarısız: cevap bağlamında “retrieval unavailable” durumu görünür.
- 20.480 token aşımı: öncelikle eski konuşma özetlenir; sistem mesajı veya son
  kullanıcı mesajı sessizce kesilmez.
- Bellek kapısı aşılır: release başarısız olur; limit yükseltilerek sorun
  gizlenmez.

## 11. Test ve Doğrulama

### 11.1 Unit

- 98M parametre hesabı.
- Attention pattern ve KV head doğrulaması.
- 20.480 toplam token bütçesi.
- 15.360 özetleme tetikleyicisi.
- 2.048 retrieval sınırı.
- Özet transaction ve kaynak mesaj kimlikleri.
- Cache açık/kapalı generation eşitliği.
- 100M→200M dönüşüm mapping ve shape testleri.

### 11.2 Integration

- Transformers save/load ve `AutoModelForCausalLM`.
- Dynamic/Static cache generation.
- INT8 CPU quantized yükleme.
- Discord/Instagram adaptörü olmadan fake transport ile CLI araç sözleşmesi.
- Gerçek mesaj deposunda yetki filtreli arama.
- Colab TPU 1, 10, 100 ve 1.000 step checkpoint/resume smoke.

### 11.3 Model eval

- Türkçe ve Almanca perplexity/QA.
- Türkiye, Almanya ve dünya tarihi.
- Matematik ve mantık.
- Uzun bağlam needle/passkey/retrieval.
- Kimlik tutarlılığı.
- Abstention ve kaynak sadakati.
- Prompt echo, mojibake, low-information ve role-boundary regresyonları.

### 11.4 Donanım benchmark

Kullanıcının i3-1215U/8 GB makinesinde ayrı süreçle:

- Soğuk yükleme süresi.
- 2K/8K/15K/20K prefill süresi.
- Token/s.
- Peak RSS ve USS.
- Model ağırlığı, KV cache ve temporary tahminleri.
- 128, 512 ve 2.048 token üretim profilleri.

## 12. Birincil Referanslar

- Hugging Face Transformers cache:
  https://huggingface.co/docs/transformers/kv_cache
- Hugging Face RoPE seçenekleri:
  https://huggingface.co/docs/transformers/internal/rope_utils
- Hugging Face TorchAO quantization:
  https://huggingface.co/docs/transformers/quantization/torchao
- PyTorch x86 INT8:
  https://pytorch.org/blog/int8-quantization/
- bert2BERT:
  https://arxiv.org/abs/2110.07143
- Masked Structural Growth:
  https://arxiv.org/abs/2305.02869
- SOLAR depth up-scaling:
  https://arxiv.org/abs/2312.15166
- FineWeb2:
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- FineWeb2-HQ:
  https://huggingface.co/datasets/epfml/FineWeb2-HQ

