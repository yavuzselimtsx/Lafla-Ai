# Ücretsiz Veri Kaynağı Seçimi

Bu not LaflaGPT 400M Thinking için ücretsiz ve açık erişimli kaynakları nasıl
kullanacağımızı belirler. Amaç veri çeşitliliğini artırmak, fakat lisans veya
kalite belirsizliğini eğitim hattına sessizce sokmamaktır.

## Ana Gövde

- FineWeb2 `tur_Latn`: Türkçe pretraining ana gövdesi. Lisans ODC-By 1.0,
  dil-script alt kümesi var, filtreleme/dedup/PII süreci belgeli.
  Kaynak: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- FineWeb2-HQ `tur_Latn`: kalite ağırlığı için ikinci gövde. ODC-By lisanslı,
  Türkçe alt klasörü mevcut.
  Kaynak: <https://huggingface.co/datasets/epfml/FineWeb2-HQ>
- HPLT v2 cleaned Turkish: CC0 paketleme lisansı güçlü, fakat kaynak metin
  hakları ve bölgesel uyum ayrıca gözden geçirilir.
  Kaynak: <https://hplt-project.org/datasets/v2.0>
- Wikimedia Türkçe: formatı temiz, bilgi yoğunluğu yüksek; CC-BY-SA/GFDL
  atıf/paylaşım koşulları nedeniyle ayrı takip edilir.
  Kaynak: <https://huggingface.co/datasets/wikimedia/wikipedia>

## Post-Training

- Aya Turkish: instruction tuning için öncelikli kaynak. Türkçe örnek ve test
  kapsamı mevcut.
  Kaynak: <https://huggingface.co/datasets/CohereLabs/aya_dataset>
- Tulu 3 SFT mixture: güçlü post-training reçetesi; fakat alt veri setlerinde
  farklı ve yer yer non-commercial koşullar bulunduğu için yalnız reviewed
  subset ile kullanılır.
  Kaynak: <https://huggingface.co/datasets/allenai/tulu-3-sft-mixture>
- Lafla identity dialogues: Lafla kimliği, Türkçe üslup, geliştirici modu ve
  ürün davranışı için owned veri.

## Red Lines

- Lisansı bilinmeyen kaynak eğitim shardına girmez.
- Synthetic veri öğretmen modeli, kaynak ve üretim tarihi olmadan girmez.
- `review_required` kaynaklar otomatik downloader ile çekilmez.
- Test splitleri pretraining'e karıştırılmaz.
- Kaynak URL ve source_id rapora yazılmadan checkpoint kabul edilmez.

Uygulanan plan: `configs/data/lafla-400m-source-plan.json`.
