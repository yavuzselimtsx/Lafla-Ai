# Lafla AI Çalışma Alanı

Lafla AI, Lafla için Türkçe öncelikli ve sıfırdan eğitilecek decoder-only
transformer çalışma alanıdır. Hazır GPT-2, Llama, Mistral, Qwen veya benzeri
model ağırlığı kullanılmaz. Kimlik davranışı mimariye gömülmez; kimlik
yapılandırması, talimat verisi, tercih verisi, sistem yönergesi ve sunum
politikası birlikte yönetilir.

Bu klasör monoreponun dışında tutulur. Monorepodaki mobil/backend kodu sızsa
bile eğitim yardımcıları, kimlik kuralları ve model deneyleri ayrı kalır.

## İlk Kontroller

```bash
python colab/lafla_1b_sifirdan_egitim.py --smoke
python -m py_compile kaynak/lafla_ai/çekirdek/model_yapılandırması.py
python -m py_compile kaynak/lafla_ai/eğitim/aşama_planlayıcı.py
python -m py_compile kaynak/lafla_ai/güvenlik/politika.py
```

Testler kaynak kod klasöründe değildir:

```bash
python -m pytest testler
```

## Kalite Çizgisi

- Hazır model ağırlığı yok.
- Lisansı belirsiz veri yok.
- PII temizlenmeden talimat veya tercih eğitimi yok.
- Lafla AI kimliği sistem promptu, eğitim verisi ve cevap politikasıyla korunur.
- Eğitim aşamaları `ön_eğitim`, `talimat_uyumu`, `tercih_optimizasyonu` ve
  `değerlendirme` olarak ayrıdır.
- Gösteriş amaçlı demo modüller üretim sayılmaz; her modül ya doğrulama, ya
  veri denetimi, ya eğitim, ya güvenlik, ya da sunum görevine bağlıdır.
