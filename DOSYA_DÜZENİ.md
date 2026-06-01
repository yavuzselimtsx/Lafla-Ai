# Lafla AI Dosya Düzeni

Bu çalışma alanı monoreponun dışında durur. Amaç birkaç JSON ve tek Colab
betiği bırakmak değil; veri, model çekirdeği, eğitim, değerlendirme, güvenlik
ve sunum katmanlarını ayrı sorumluluklarla yönetmektir.

```text
Lafla ai/
  README.md
  DOSYA_DÜZENİ.md
  belgeler/
    mimari-kararlar.md
    eğitim-aşamaları.md
  colab/
    lafla_1b_sifirdan_egitim.py
    lafla_1b_sifirdan_egitim.ipynb
  konfigurasyon/
    lafla-1b-model.json
    lafla-ai-kimlik.json
    eğitim-hattı.json
  kaynak/
    lafla_ai/
      çekirdek/
        model_yapılandırması.py
      veri/
        veri_kataloğu.py
      eğitim/
        aşama_planlayıcı.py
        dpo.py
      değerlendirme/
        ölçütler.py
      güvenlik/
        politika.py
      sunum/
        cevap_politikası.py
        sohbet_motoru.py
  src/
    lafla_dataset.py
    lafla_model.py
    lafla_persona.py
    lafla_tokenizer.py
    veri_hazirla.py
  testler/
    lafla_ai_kalite_test.py
  veri/
    veri_manifesti.json
```

## Katman Kuralları

- `çekirdek`: Yalnız model mimarisi ve parametre doğrulaması. Persona, ürün
  metni, veri temizliği veya API mantığı içermez.
- `veri`: Lisans, PII temizliği, dil dağılımı ve token bütçesi denetimi.
- `eğitim`: Ön eğitim, talimat uyumu, DPO ve kalite kapıları ayrı aşamalar.
- `değerlendirme`: Türkçe kalite, kimlik tutarlılığı, güvenlik ve görev başarısı.
- `güvenlik`: Sistem promptuna güvenmeden giriş/çıkış politika denetimi.
- `sunum`: Sohbet durumunu, üretim girdisini ve cevap kabul politikasını yönetir.

## Şu Anki Gerçek Durum

Bu klasör üretim model ağırlığı üretmiş sayılmaz. Eklenen modüller eğitim hattını
denetlenebilir hale getirir: mimari ayarları doğrulanır, veri kaynağı lisansı
bilinmeden eğitim başlatılmaz, DPO kaybı ayrıdır, cevap güvenliği model cevabına
bırakılmaz ve testler kaynak koddan ayrı tutulur.
