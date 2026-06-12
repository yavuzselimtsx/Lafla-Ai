# Kaggle Safe Speed Profile Design

## Goal

Speed up LaflaGPT Mini 100M pretraining on Kaggle T4 x2 without changing the
6B-token target, optimizer-step token budget, learning-rate schedule, checkpoint
format, model weights, or fallback support for a single CUDA device.

## Current Bottlenecks

- The 2K and 4K curriculum stages use activation checkpointing despite ample
  T4 memory headroom. This recomputes every decoder block during backward.
- Local attention uses a Python chunk loop even when the active sequence fits
  entirely inside the 4096-token sliding window. In that case local causal
  attention is equivalent to ordinary causal attention and can use CUDA SDPA.
- T4 x2 currently processes one sequence per GPU for 16 accumulation rounds.
  The same 32-sequence optimizer batch can use two sequences per GPU and eight
  accumulation rounds when memory permits.

## Design

### Attention Fast Path

`GroupedQueryAttention._chunked_local_attention` will call PyTorch scaled
dot-product attention when `sequence_length <= sliding_window`. Longer
sequences retain the existing bounded local-attention implementation. The fast
path preserves causal visibility and dropout behavior.

### Runtime Gradient Checkpointing

The model will expose a runtime checkpointing toggle that can only disable a
feature enabled by the model config. The training profile will disable
checkpointing at 2048 tokens and automatically re-enable it when the curriculum
reaches 4096 tokens. This changes activation storage only, not model weights or
checkpoint compatibility.

### Preserved Optimizer Batch

The training config will define an optional per-CUDA-device micro batch and a
target number of sequences per optimizer step:

- T4 x2: global micro batch 4, accumulation 8, total 32 sequences.
- Single GPU: global micro batch 2, accumulation 16, total 32 sequences.
- Existing profiles without these fields preserve current behavior.

The health log will record resolved micro batch, accumulation, total sequences
per optimizer step, and runtime checkpointing state.

## Safety And Recovery

- Existing `lafla-step-*` checkpoints remain loadable.
- Cumulative token accounting remains 65,536 tokens per optimizer step during
  the 2K stage.
- If the larger micro batch runs out of memory on a particular Kaggle image,
  the user can remove the two optional batch fields and resume from the same
  checkpoint.
- `torch.compile` is intentionally excluded because DataParallel plus dynamic
  curriculum shapes is a higher-risk change.

## 70M Alternative

A 70M model would require a new architecture and training from step zero.
Existing 100M checkpoints cannot be narrowed without a separate distillation or
pruning project. It would reduce weight memory and compute, but would also
reduce multilingual capacity, reasoning quality, and long-context robustness.
The current 100M run therefore remains the recommended path.
