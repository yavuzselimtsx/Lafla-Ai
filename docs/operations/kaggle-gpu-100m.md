# Kaggle GPU 100M Run

Use `GPU T4 x2` when Kaggle offers it. The training config has `data_parallel: auto`, so two CUDA devices use PyTorch DataParallel and one GPU/P100 falls back to a normal single-device run.

When DataParallel is enabled, the runner raises the effective micro batch to at
least the CUDA device count. This keeps both T4 cards fed without increasing the
per-GPU micro batch above 1. On a single GPU the configured micro batch stays
unchanged, so P100/T4 single-device runs do not get an accidental memory bump.

## Notebook Setup

```bash
cd /kaggle/working
git clone https://github.com/yavuzselimtsx/Lafla-Ai.git LaflaAi-Core
mkdir -p /kaggle/working/LaflaAI100M/data
```

Put real training files here:

```text
/kaggle/working/LaflaAI100M/data/train.jsonl
/kaggle/working/LaflaAI100M/data/veri_manifesti.json
```

Do not start training until both files are present and non-empty. The launcher refuses to generate bootstrap/fake data.

Post-training datasets under `datasets/post_training/` are validated before the
run, but they are never passed to `train_pretrain --data-jsonl`. The pretraining
CLI and runner reject those paths fail-closed if they are added by mistake.

## Start Training

```bash
cd /kaggle/working/LaflaAi-Core
bash scripts/kaggle/start_gpu_100m.sh
```

Background run:

```bash
cd /kaggle/working/LaflaAi-Core
nohup bash scripts/kaggle/start_gpu_100m.sh > /kaggle/working/LaflaAI100M/reports/kaggle-gpu-nohup.log 2>&1 &
tail -f /kaggle/working/LaflaAI100M/reports/kaggle-gpu-launch.log
```

Resume from a checkpoint:

```bash
RESUME_FROM=/kaggle/working/LaflaAI100M/checkpoints/lafla-step-000500 \
  bash scripts/kaggle/start_gpu_100m.sh
```

## Output Layout

```text
/kaggle/working/LaflaAI100M/
  data/
  tokenizer/
  checkpoints/
  reports/
  hf-package/
  archives/
```

Final archive:

```text
/kaggle/working/LaflaAI100M/archives/lafla-100m-thinking-kaggle-gpu-run.tar.gz
```
