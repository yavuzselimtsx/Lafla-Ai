# Kaggle GPU 100M Run

Use `GPU T4 x2` when Kaggle offers it. The launcher uses `torchrun` with one
DDP process per CUDA device. A one-GPU/P100 session uses the same runner in
single-device mode.

Profiles without explicit CUDA batch tuning raise the effective micro batch to
at least the CUDA device count. Single-GPU and non-CUDA runs keep their
configured legacy batch behavior.

The safe-speed profile overrides that legacy fallback with an explicit
per-device CUDA micro batch while preserving 32 sequences per optimizer step:

```text
T4 x2: 2 per rank x 2 ranks x 8 accumulation = 32 sequences
T4/P100 x1: 2 per process x 1 rank x 16 accumulation = 32 sequences
```

DDP uses `no_sync()` for the first seven accumulation microsteps and performs
gradient reduction only on the eighth. Rank zero alone writes logs and atomic
checkpoints. Existing plain and DataParallel checkpoints need no conversion.

Activation checkpointing is disabled for the 2048 curriculum stage and
re-enabled at 4096 tokens. Local attention uses CUDA SDPA while the complete
sequence fits inside the 4096-token sliding window. Neither optimization changes
checkpoint weights or the 65,536-token optimizer-step budget at 2048 context.

If a particular Kaggle image reports CUDA out of memory, remove
`cuda_micro_batch_size_per_device` and
`target_sequences_per_optimizer_step` from
`configs/training/kaggle/kaggle-gpu-100m.yaml`, then resume from the same
checkpoint. The zero/default behavior retains the older conservative batch.

Fused AdamW, pinned non-blocking transfer, and native SDPA GQA are capability
gated. Unsupported Kaggle/PyTorch builds retain the standard AdamW, blocking
transfer, or explicit KV expansion paths as appropriate.

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

Do not start training until both files are present and non-empty. The launcher
refuses to generate bootstrap/fake data. It also streams through every JSONL
record before tokenizer/training startup; malformed, truncated, empty, or
mojibake records fail before GPU allocation.

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

Before committing to the long run, resume for 20-to-50 steps and verify:

```text
parallelism = distributed_data_parallel
world_size = 2
per_process_micro_batch_size = 2
effective_micro_batch_size = 4
effective_gradient_accumulation_steps = 8
sequences_per_optimizer_step = 32
```

At 2048 context, cumulative tokens must increase by `65,536` per optimizer
step. Loss and gradient norms must remain finite, and a newly written
checkpoint must pass the existing checkpoint contract.

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
