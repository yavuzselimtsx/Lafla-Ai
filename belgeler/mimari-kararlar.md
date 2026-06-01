# Lafla AI Mimari Kararları

Lafla AI, hazır bir modelin adını değiştirerek oluşturulmayacak. Model sıfırdan başlatılır; yetenek kazanımı ise veri kalitesi, tokenizer, ön eğitim, instruction tuning, tercih optimizasyonu ve değerlendirme kapılarından gelir.

## Kararlar

- Decoder-only transformer korunur; çünkü sohbet ve kod üretimi için en doğrudan eğitim hedefi budur.
- RoPE, RMSNorm, SwiGLU ve grouped-query attention kullanılır; bunlar uzun bağlam ve verimli çıkarım için pratik modern bloklardır.
- Kimlik “Ben Lafla AI’yım” gibi sabit cevaplarla değil, sistem yönergesi ve instruction kayıtlarıyla öğretilir.
- Güvenlik davranışı tamamen modele gömülmez. Çalışma zamanı sistem yönergesi ve politika katmanı değiştirilebilir kalır.
- SFT ve DPO ayrı aşamalardır. SFT konuşma biçimini öğretir; DPO tercih edilen cevapları tercih edilmeyenlere göre güçlendirir.
- Değerlendirme olmadan checkpoint “iyi model” sayılmaz.

## Kaynak Dayanakları

- DPO, tercih verisinden doğrudan politika optimizasyonu yaparak ayrı reward model ve RL döngüsünü azaltır.
- Yapılandırılabilir güvenlik yaklaşımı, davranışı yalnızca ağırlığa sabitlemek yerine sistem yönergesiyle kontrol edilebilir tutmayı hedefler.

## Lafla’ya Özel Kalite Kapıları

- Türkçe konuşma doğallığı
- Lafla AI kimlik tutarlılığı
- Kod cevabında hardcoded/demo öneri üretmeme
- Güvenlik ve gizlilik sınırlarını aşmama
- Cevapta belirsizliği açık söyleme
- Uzun konuşmada bağlamı tutarlı izleme
