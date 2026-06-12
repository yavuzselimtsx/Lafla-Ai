# LaflaGPT 380M Colab TPU v5e Plan

## Decision

Use Colab TPU v5e before T4. TPU v5e has 197 BF16 TFLOPs per chip with 16 GB
HBM and 800 GiB/s bandwidth. NVIDIA T4 has 65 mixed-precision TFLOPs, 16 GB
GDDR6, and 300 GB/s bandwidth. For this 380M bf16 pretraining run, TPU v5e is
the better free accelerator if PyTorch/XLA initializes correctly.

T4 remains the fallback because the existing PyTorch CUDA path is simpler and
more reliable, but it will be much slower.

## Files

- TPU config: `configs/training/colab-tpu-v5e-380m-50000.yaml`
- T4 fallback config: `configs/training/colab-t4-380m-fallback.yaml`
- TPU launcher: `scripts/colab_start_tpu_v5e_380m.sh`
- Root convenience launcher: `colab_start_tpu_v5e_380m.sh`

## Colab Flow

1. Select TPU v5e runtime in Colab.
2. Upload `LaflaAi-Core-src-colab-tpu-380m.zip` and `colab_start_tpu_v5e_380m.sh` to `/content`.
3. Run `bash /content/colab_start_tpu_v5e_380m.sh`.
4. Check `/content/LaflaAI380M/reports/colab-tpu-launch.log`.

The script installs `torch_xla[tpu]`, sets `PJRT_DEVICE=TPU`, checks the XLA
device, runs preflight, prepares real data, trains the tokenizer from a real
sample, and starts the 380M bf16 pretraining run.

## Fallback

If TPU initialization fails before training starts, switch Colab runtime to T4
and use `configs/training/colab-t4-380m-fallback.yaml`. Do not try to run the
TPU config on a T4 runtime.
