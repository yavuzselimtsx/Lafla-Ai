# Thinking Model Eğitim Notları

Bu not LaflaAi-Core içindeki thinking davranışının nasıl ele alınacağını
belirler. Amaç özel token ekleyip modelden sihir beklemek değil; pretraining,
thinking-SFT, tercih optimizasyonu ve runtime gizlilik katmanını ayrı tutmaktır.

## Yerel Referanslardan Alınan Ders

OLMo Core tarafında SFT verisi `label_mask` ile gelir; hangi tokenların loss
alacağı veri hattında açıkça taşınır. Bu yüzden Lafla'da kullanıcı, sistem ve
maskelenmesi gereken private thinking bölümleri `-100` label ile ayrılır.

GPT-NeoX tarafında chat-template preprocessing, only-last-turn ve DPO chosen /
rejected çiftleri post-training tarafında tutulur. Lafla'da da thinking SFT,
pretraining runner içine gizli branch olarak eklenmez; `post_training/` altında
ayrı sözleşme ve validator ile büyür.

## Araştırma Kararı

- Instruction tuning, temel dil modelinden farklı bir davranış katmanıdır.
  İnsan gösterimleri ve tercih verisi küçük modellerde bile davranışı güçlü
  şekilde değiştirebilir.
- Thinking davranışı için iki güvenli yol vardır: supervised thinking trace
  verisiyle SFT veya doğrulanabilir görevlerde RL/preference aşaması. Colab 400M
  hedefinde ilk güvenli adım SFT + DPO hazırlığıdır.
- Private düşünce çıktısı ürün cevabına doğrudan sızmamalıdır. DEV modda
  araştırma için görülebilir; PROD runtime `strip_thinking_for_public` benzeri
  politika katmanından geçmelidir.
- Developer research runtime profilinde ham `<|think|>` bloğu raporlanır; normal
  runtime profilinde yalnız public cevap döner.

Birincil kaynaklar:

- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- Quiet-STaR: <https://arxiv.org/abs/2403.09629>
- InstructGPT: <https://arxiv.org/abs/2203.02155>
- DPO: <https://arxiv.org/abs/2305.18290>

## Lafla Veri Sözleşmesi

Thinking SFT JSONL satırı şu alanları taşır:

```json
{"system":"...","user":"...","thinking":"...","assistant":"..."}
```

Kurallar:

- `system`, `user`, `thinking`, `assistant` boş olamaz.
- `system` ve `user` özel kontrol tokenı taşıyamaz.
- `assistant` içinde `<|think|>` veya `<|/think|>` bulunamaz; bu alan public
  cevaptır.
- `thinking` uzunluğu config sınırını aşamaz.
- Eğitim örneği render edilirken thinking bloğu assistant turn içinde yer alır,
  fakat config'e göre loss'tan segment bazlı çıkarılabilir.
- Sentetik güvenlik seed'lerinde `thinking` alanı tam gizli zincir-düşünce
  dökümü değil, kısa karar adımlarıdır. Amaç modelin jailbreak, özel veri,
  tool izni ve kaynak uydurma durumlarında hangi güvenli davranış sınıfına
  geçeceğini öğretmektir.

## Colab Öncesi Kapılar

Zorunlu:

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.quality_scan --root .
PYTHONPATH=src python -m lafla_ai_core.cli.preflight configs/model/lafla-400m-thinking.yaml configs/training/colab-t4-400m-4000.yaml configs/tokenizer/turkish-thinking-bpe.yaml configs/runtime/desktop-cpu-4bit.yaml configs/post_training/lafla-thinking-sft.yaml
```

Thinking SFT verisi varsa:

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.validate_thinking_sft \
  --input /content/LaflaAI/data/thinking_sft.jsonl \
  --report /content/LaflaAI/reports/thinking-sft-audit.json
```

Yerel sentetik seedler:

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.generate_synthetic_chat_seed \
  --profile configs/post_training/lafla-100m-seed-profile.json
PYTHONPATH=src python -m lafla_ai_core.cli.generate_safety_jailbreak_seed \
  --profile configs/post_training/lafla-100m-seed-profile.json
```

Profil dosyası model kimliği, dil odağı, senaryo metinleri ve modifierları taşır.
Generator kodu bu metinleri gömmez; 200M veya yeni dil hedefinde ayrı profil
dosyası kullanılmalıdır.

100M Colab launcher bu iki seed'i post-training girdisi olarak doğrular:

- `datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl`
- `datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl`

Bu kapılar geçmeden uzun Colab koşusu başlatılmaz.

Developer runtime kontrolü:

```bash
PYTHONPATH=src python -m lafla_ai_core.cli.preflight configs/runtime/developer-research.yaml
```
