# Lafla AI Eğitim Aşamaları

## 1. Veri Hazırlama

Veri manifesti kaynak, lisans, kullanım amacı ve dil bilgisini taşır. PII temizliği ve tekrar ayıklama yapılmadan eğitim shardı üretilemez.

## 2. Tokenizer Eğitimi

Tokenizer hazır modelden alınmaz. Türkçe ek yapısı, kod blokları, URL benzeri parçalar ve sohbet özel tokenları aynı sözlükte öğrenilir.

## 3. Ön Eğitim

Amaç metin devam ettirme becerisini kazandırmaktır. Bu aşamada model Lafla AI gibi konuşmayı tam öğrenmiş sayılmaz.

## 4. Instruction Tuning

Kullanıcı/asistan kayıtlarıyla sohbet biçimi, Lafla kimliği, Türkçe üslup ve kod kalitesi öğretilir.

## 5. Tercih Optimizasyonu

DPO veri seti `prompt`, `chosen`, `rejected` alanlarını taşır. Tercih edilen cevaplar daha yararlı, daha güvenli ve Lafla kurallarına daha uyumlu olmalıdır.

## 6. Değerlendirme

Her checkpoint aşağıdaki kapılardan geçer:

- Kimlik tutarlılığı
- Türkçe cevap kalitesi
- Kod kalitesi
- Güvenlik sınırları
- Halüsinasyon direnci
- Kısa ve uzun bağlam tutarlılığı

## 7. Sunum

Sunum katmanı sistem yönergesini, sıcaklık ayarını, maksimum çıktı uzunluğunu ve güvenlik politikasını uygular. Model tek başına ürün güvenliği sayılmaz.
