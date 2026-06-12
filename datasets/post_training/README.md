# Post-Training Datasets

Bu alan pretraining corpus degildir. SFT/davranis/safety seedleri burada
kategorize edilir.

```text
thinking/jsonl/      thinking ve belirsizlik davranisi JSONL seedleri
thinking/manifests/  thinking seed manifestleri
safety/jsonl/        jailbreak ve policy direnclilik JSONL seedleri
safety/manifests/    safety seed manifestleri
```

Launcher'lar bu dosyalari dogrular, ancak `train_pretrain --data-jsonl`
listesine eklemez.
