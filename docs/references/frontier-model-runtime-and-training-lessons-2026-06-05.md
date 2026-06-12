# Frontier Model Runtime and Training Lessons - 2026-06-05

Bu not OpenAI, Anthropic, Google Gemini ve Hugging Face kaynaklarindan LaflaAi-Core'a clean-room olarak alinacak dersleri toplar. Kapali modellerin ic mimarisi kamuya acik degilse mimari iddia uydurulmaz.

## Kaynaklar

- OpenAI GPT-5.5 docs: `https://developers.openai.com/api/docs/guides/latest-model.md`
- OpenAI Harmony/gpt-oss format: `https://cookbook.openai.com/article/harmony/`
- Anthropic Claude Opus 4.8 announcement: `https://www.anthropic.com/news/claude-opus-4-8`
- Anthropic Claude Opus page: `https://www.anthropic.com/claude/opus`
- Anthropic extended thinking docs: `https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking`
- Google Gemini thinking docs: `https://ai.google.dev/gemini-api/docs/thinking`
- Google Gemini structured output: `https://ai.google.dev/gemini-api/docs/structured-output`
- Google Gemini text generation/system instructions: `https://ai.google.dev/gemini-api/docs/text-generation`
- Hugging Face chat templates: `https://huggingface.co/docs/transformers/chat_templating`
- Hugging Face custom models: `https://huggingface.co/docs/transformers/custom_models`

## Dogrulanmis Dersler

1. Reasoning kalitesi yalniz prompt metni degildir. Runtime tarafinda effort, verbosity, state, tool contract ve output schema ayri sozlesmeler olarak tutulmalidir.
2. Mesaj formatlari model davranisinin parcasi sayilir. `system`, `user`, `assistant`, private thinking ve final cevap tek merkezden render edilmelidir.
3. Public cevap ile private thinking ayrimi veri, SFT label mask ve runtime policy tarafinda birlikte korunmalidir.
4. Halusinasyon azaltma tek bir sampling ayariyla cozulemez. Release gate, kaynak/citation kontrolu, belirsizlik kabul davranisi ve grounded eval setleri gerekir.
5. Hugging Face'te kullanilabilirlik icin `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, model card ve chat template birlikte yayinlanmalidir.
6. Ozel mimari agirliklar standart `AutoModelForCausalLM` ile calismaz. Ya standart mimariye uyumlu agirlik donusumu gerekir ya da model repo kendi modelleme dosyalarini tasir ve kullanici `trust_remote_code=True` ile yukler.
7. Structured outputs ve schema validation egitim verisi hazirlama, eval raporu ve release kararlarinda uygulama katmani zorunlulugu olarak modellenmelidir.

## LaflaAi-Core Kararlari

- `tokenizer/chat_template.py` role ve thinking token sozlesmesinin tek kaynagi olur.
- Runtime public output `clean_decoded_text` hattindan gecmeden kullaniciya donmez.
- Checkpoint `READY.json` yalniz var diye guvenilir sayilmaz; format ve zorunlu dosyalar dogrulanir.
- Colab final arsivinden once hashli artifact manifest uretilir.
- HF paketleme komutu tokenizer/runtime metadata dosyalarini yazar; standart mimari uyumlulugu iddia etmez.
- `evaluation/grounding.py` desteksiz factual cevabi release kapisinda kirmizi yapar.
- `runtime/generation_contract.py` prompt ids, stop ids ve completion-only decode
  sinirini tek sozlesmede toplar.
- `evaluation/runtime_regressions.py` ekran goruntusundeki prompt echo, mojibake
  ve role-boundary hatalarini deterministik gate case'lerine cevirir.
- `data/mixture.py` manifest kaynak agirliklarini normalize sample budget planina
  cevirir.
- `export/hf_remote_code.py` HF reposuna self-contained Lafla config/model code
  yazilmasini saglar; bu, `trust_remote_code=True` yukleme yolunun temelidir.

## Acik Kalan Buyuk Isler

- Standart Llama benzeri HF agirlik donusumu: mevcut `qk_norm=true` ve agirlik isimleri nedeniyle ayri donusum plani ister.
- Gercek hallucination eval seti: kaynakli Turkce QA, kimlik cevaplari, lisans sorulari, kod/dokuman ground truth ornekleri gerekir.
- Tokenizer kalite deneyleri: yeni chat template ile tokenizeri bastan egitip roundtrip, unknown-rate, byte-level yuzey, Turkce ek/kok parcalanmasi ve kisa cevap smoke uretimi olculmelidir.
