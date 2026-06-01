# Lafla AI Mimari Kararları

Lafla AI, hazır bir modelin adını değiştirerek oluşturulmayacak. Model
sıfırdan başlatılır; yetenek kazanımı veri kalitesi, tokenizer, ön eğitim,
talimat uyumu, tercih optimizasyonu ve değerlendirme kapılarından gelir.

## Model Çekirdeği

- Decoder-only transformer korunur; sohbet ve kod üretimi için doğrudan eğitim
  hedefi sağlar.
- RoPE, RMSNorm, SwiGLU ve grouped-query attention kullanılır. Bu bloklar uzun
  bağlam, kararlı eğitim ve daha verimli çıkarım için pratik modern seçimlerdir.
- Kimlik “Ben Lafla AI’yım” gibi sabit cevaplarla değil; kimlik verisi, sistem
  yönergesi, talimat örnekleri ve konuşma kalite politikasıyla öğretilir.
- Güvenlik davranışı yalnızca ağırlıklara gömülmez. Çalışma zamanı politika
  katmanı ve halüsinasyon kapıları ayrı kalır.

## Eğitim Hattı

1. Veri denetimi: lisans, PII, kaynak URL, güven seviyesi.
2. Tokenizer: Türkçe karakterleri ve Lafla teknik metinlerini doğru kapsayan
   sözcük parçası dağılımı.
3. Ön eğitim: FineWeb2/FineWeb2-HQ Türkçe ağırlıklı temel dil öğrenimi.
4. Talimat uyumu: Aya ve Lafla kimlik konuşmalarıyla doğal Türkçe cevap.
5. Tercih optimizasyonu: DPO ile daha doğru, daha az uyduran cevap tercihi.
6. Halüsinasyon değerlendirme: TruthfulQA, RAGTruth, RAGTruth-TR ve Lafla yerel
   konuşma testleri.

## Halüsinasyon Kararı

İlk eğitimde sıfır halüsinasyon garanti edilemez. Bu yüzden hedef “model asla
uydurmayacak” iddiası değil; kanıt yokken kesin konuşmayan, belirsizliği doğru
söyleyen ve desteklenmeyen iddia oranı ölçülen bir sistem kurmaktır. Checkpoint
ancak `halüsinasyon-kapıları.json` eşiklerini geçerse kabul edilir.

## Kaynak Dayanakları

- FineWeb2: çok dilli, ODC-By lisanslı ve belgeli ön eğitim kaynağı.
- FineWeb2-HQ: FineWeb2’den türetilmiş yüksek kalite alt küme.
- Aya Dataset: çok dilli instruction tuning için açık veri.
- Tülu 3: açık post-training reçetesi; SFT, preference data ve RL/verifiable
  reward yaklaşımıyla iyi bir referans.
- TruthfulQA ve RAGTruth: doğruluk ve halüsinasyon ölçümü için değerlendirme
  hattına alınır.
