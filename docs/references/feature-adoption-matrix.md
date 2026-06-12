# Feature Adoption Matrix

Bu tablo OLMo Core ve GPT-NeoX'ten hangi ozellik siniflarinin Lafla'ya temiz
oda olarak alinacagini belirler.

| Alan | OLMo Core dersi | GPT-NeoX dersi | LaflaAi-Core karari |
| --- | --- | --- | --- |
| Config | Dataclass tabanli, ic ice config, override | Cok kapsamli YAML argumanlari | Typed schema + fail-closed preflight |
| Data | Composable source/mix mantigi | Packed/mmap dataset secimleri | Manifest kaynak katalogu + shard raporu + packing |
| Tokenizer | Data katmanindan ayrim | Vocab/merge path disiplini | Turkce roundtrip ve mojibake gate |
| Training | Trainer, callbacks, monitors | Train impl secimi, ZeRO, BnB, checkpoint | Stage runner + callback registry + Colab profile |
| Checkpoint | Config saver, checkpointer, metric saver | keep_last, extra save, load/save flags | Atomic checkpoint + artifact manifest + retention |
| Eval | Task groups, metrics, evaluator | eval interval/iters | Gate suite: TR, identity, safety, hallucination |
| Generation | Sampling ve chat ayrimi | generation config | Runtime policy + sampling profile + DEV/PROD split |
| Post-training | SFT script family + label mask | dpo/rm/kto train impl + only-last chat preprocessing | Thinking SFT/DPO/KTO hazir sozlesme, once SFT sonra DPO |
| Thinking | Reasoning ve SFT script ayrimi | Chat-template ve tercih veri ciftleri | Private thinking segmentleri maskelenebilir, public cevap ayri kalir |
| Distributed | DP/TP/PP/CP/EP ayrimi | DeepSpeed runner | Colab single GPU first, distributed adapter later |
| Low power | Export/generation ayrimi | quantization/runtime configs | 4-bit CPU package is first-class |

## Alinacak Ozellik Aileleri

- Typed config ve override disiplini.
- Callback tabanli training observer sistemi.
- Checkpoint retention, atomic write, metric saver.
- Eval task group ve release gate kavrami.
- Data source mixture ve weighted sampling kavrami.
- SFT/DPO/KTO icin ayri train implementation sozlesmesi.
- Thinking SFT icin JSONL validator, assistant-only label mask ve public strip kapisi.
- CPU/dusuk guc export yolu.

## Ertelenecek Ama Tasarimda Yer Ayrilacak Ozellikler

- Multi-node distributed training.
- MoE.
- Pipeline/tensor/context parallel.
- Float8.
- Online RL.
- Autotuning.

Bu ozellikler klasor ve config semasinda yer bulur, ama Colab 400M hedefini
kirletmeyecek sekilde kapali gelir.
