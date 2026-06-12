# LLM Egitim Kaynagi Inceleme Notu

Bu not `OLMo-core-main` ve `gpt-neox-main` kaynaklarinin LaflaAi-Core icin nasil
kullanilacagini belirler.

## OLMo Core

Gozlenen guclu taraflar:

- Egitim, eval, optim, distributed ve generation alanlari ayrilmis.
- Config merkezli dusunce daha temiz.
- Arastirma ve resmi egitim scriptleri birbirinden ayriliyor.
- Test ve internal yardimci katmanlari var.

Lafla icin ders:

- `training`, `evaluation`, `optim`, `runtime`, `data` ayrimi korunmali.
- Notebook yerine moduler script ve config sozlesmesi esas olmali.
- Preflight ve health log resmi egitim arayuzunun parcasi olmali.
- SFT ve reasoning calismalarinda loss kapsami veri tarafindan `label_mask`
  benzeri acik sozlesmeyle tasinmali.
- Think/Instruct SFT script aileleri pretraining scriptinden ayri kalmali.
- `get_labels()` benzeri katmanlar `label_mask`, `attention_mask` ve
  `instance_mask` ile kaybi açıkça kapatıyor; Lafla'da `-100` label politikası
  bunun temiz oda karşılığıdır.

## GPT-NeoX

Gozlenen guclu taraflar:

- Buyuk model egitiminde config zenginligi.
- Checkpoint, dataset ve post-training araclari ayrilmis.
- Distributed egitim mantigi net bir ana komut etrafinda toplaniyor.

Lafla icin ders:

- Colab T4 kucuk hedef olsa bile dosya/agac disiplini buyuk model gibi olmali.
- SFT/DPO post-training ana pretraining koduna karistirilmamali.
- Checkpoint donusum/export araclari ilk gunden tasarlanmali.
- Chat-template preprocessing, only-last assistant loss ve DPO chosen/rejected
  ciftleri post-training veri katmaninda modellenmeli.
- Generation/chat tarafı eğitim döngüsünün içine gömülmemeli; runtime policy ve
  render katmanı ayrı kalmalıdır.

## 2026-06-05 Ek Uygulama Kararlari

- OLMo generation dokumanindaki completion-only yaklasimi Lafla'da
  `runtime/output_guard.py` ile runtime sinirina tasindi; prompt tokenlari
  public cevaba yeniden decode edilmemeli.
- GPT-NeoX chat-template preprocessing ve `only-last`/assistant loss-mask
  disiplini Lafla'da `post_training/thinking_sft.py` ve
  `training/phase_plan.py` sozlesmeleriyle temsil edilir.
- GPT-NeoX stop-token davranisinin Lafla karsiligi `ROLE_STOP_SEQUENCES` ve
  `role_boundary_stop` gate'idir; `<|user|>`, `<|system|>`, ikinci
  `<|assistant|>` veya `<|eos|>` public cevaba gecmemelidir.
- `configs/evaluation/release-gates.yaml` artik `prompt_echo_guard`,
  `mojibake_decode` ve `role_boundary_stop` kapilarini zorunlu sayar.
- `training_phase_plan` CLI'si tokenizer audit, base pretrain, anneal midtrain,
  instruction SFT, thinking SFT ve release eval asamalarini JSON olarak verir.
- OLMo `generate_batch(..., completions_only=True)` dersi artik token-id
  seviyesinde `runtime/generation_contract.py` ile temsil edilir; prompt tokenlari
  decode edilen public completion'a dahil edilmez.
- GPT-NeoX stop-token ve weighted dataset dersi `evaluation/runtime_regressions.py`
  ile release case'lerine, `data/mixture.py` ile normalize sample budget planina
  ayrildi.
- Hugging Face paketleme artik opsiyonel `model_config` aldiginda `config.json`,
  `configuration_lafla.py` ve `modeling_lafla.py` yazar; bu, agirliklar HF'ye
  kondugunda kullanicinin yerel LaflaAi-Core reposuna mecbur kalmamasinin
  baslangic sozlesmesidir.

## Temiz Oda Siniri

Bu kaynaklardan:

- kod kopyalanmaz,
- test fixture alinmaz,
- lisans metni tasinmaz,
- dosya agaci aynen klonlanmaz.

Yalniz sorumluluk ayrimi, operasyonel dersler ve mimari desenler Lafla'ya kendi
diliyle yeniden ifade edilir.
