# Artifacts

Bu klasore buyuk model dosyalari commit edilmez.

Beklenen artifact aileleri:

- `checkpoints/`
- `generated_datasets/`
- `exports/`
- `reports/validation/`
- `tokenizers/`

Büyük dosyalar Drive, harici disk veya artifact storage üzerinde tutulur. Repo
içinde yalnız küçük denetim seedleri, hash, rapor ve yeniden üretim komutları
saklanır. `generated_datasets/` Git tarafından izlenmez.
