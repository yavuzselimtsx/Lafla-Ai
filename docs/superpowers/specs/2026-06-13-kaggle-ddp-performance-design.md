# Kaggle DDP Performance Design

## Goal

Improve LaflaGPT Mini 100M pretraining throughput on Kaggle T4 x2 without
changing model parameters, training data, optimizer-step batch size,
learning-rate semantics, cumulative-token accounting, checkpoint format, or
single-GPU support.

The current `lafla-step-*` checkpoint must remain directly resumable. The
optimization is accepted only when a short Kaggle comparison shows stable loss,
gradient norms, token accounting, and checkpoint recovery.

## Non-Goals

- Do not reduce the model below 100M parameters.
- Do not change the curriculum, target token count, tokenizer, or data mixture.
- Do not introduce `torch.compile`, FSDP, tensor parallelism, or a new
  checkpoint format.
- Do not embed device names, Kaggle paths, model identity text, or behavior
  policy in training code.
- Do not silently continue with a partially initialized distributed process
  group.

## Architecture

### Launch Selection

The Kaggle launcher selects its execution mode from detected CUDA capacity:

- Two or more CUDA devices: launch the training CLI with `torchrun`, one process
  per selected GPU.
- One CUDA device: launch the existing single-process training CLI.
- No CUDA device: retain the existing fail-closed Kaggle GPU preflight.

The runner detects a distributed launch from standard `torchrun` environment
variables. It does not infer ranks from GPU product names. Distributed setup and
cleanup live in a focused training module rather than the shell script or model
class.

### Distributed Training

The multi-GPU path uses `DistributedDataParallel` with NCCL and one exclusive
CUDA device per process. The plain model is wrapped only after construction and
checkpoint restore, preserving existing state-dict keys.

Gradient accumulation uses `no_sync()` for every non-final microstep. Gradient
reduction occurs only on the final microstep before clipping and the optimizer
step. With the current T4 x2 profile this reduces distributed gradient
synchronization from eight reductions per optimizer step to one.

The global optimizer batch remains:

```text
2 ranks x 2 sequences per rank x 8 accumulation = 32 sequences
```

At the 2048-token curriculum stage, cumulative token accounting remains:

```text
32 sequences x 2048 tokens = 65,536 tokens per optimizer step
```

### Data Sharding

Each rank receives a deterministic, non-overlapping subset of training blocks.
Rank sharding happens after the existing deterministic block stream is defined,
so rank zero consumes positions `0, world_size, ...` and rank one consumes
positions `1, world_size + 1, ...`.

All ranks execute the same number of optimizer steps. Resume retains the
existing restart-stream policy and records rank/world-size diagnostics. The
design does not claim exact data-position continuation because the current
checkpoint deliberately restarts the stream instead of performing a slow skip.

### Checkpoint And Logging

Only rank zero writes checkpoints, health logs, reports, and final artifacts.
All ranks synchronize before and after checkpoint publication. A rank failure
aborts the distributed job instead of allowing another rank to publish an
incomplete checkpoint.

Checkpoint payloads continue to contain the unwrapped model state, optimizer
state, RNG state, model config, and training state. Existing DataParallel and
plain checkpoints remain loadable by the DDP path, and DDP checkpoints remain
loadable by single-GPU inference and training.

## Additional Safe Fast Paths

### Fused AdamW

CUDA training requests fused AdamW only when the installed PyTorch build accepts
it for the active parameter dtype and device. Unsupported builds fall back to
the existing optimizer construction. Optimizer hyperparameters and checkpoint
state semantics remain unchanged.

### Host-To-Device Transfer

CUDA DataLoaders may use pinned host memory and batches use non-blocking device
transfer. CPU and unsupported environments retain blocking transfer. These
settings affect transfer scheduling only.

### Label Reuse

The causal language-model target reuses the input token tensor instead of
cloning it when the loss implementation does not mutate labels. This removes an
unnecessary device allocation without changing target values.

### Native GQA

Attention may request SDPA native grouped-query attention when the installed
PyTorch/CUDA backend supports it and the configured query/KV head relationship
is valid. Any unsupported backend or runtime rejection falls back to the
existing explicit KV-head expansion. Numerical routing tests cover both paths.

## Configuration

Distributed and transfer behavior is configuration-driven. Defaults preserve
existing non-Kaggle profiles. The Kaggle profile enables the safe automatic
policy without referencing T4 product names in Python:

- distributed mode: automatic
- distributed backend: automatic, resolving CUDA to NCCL
- synchronize gradients: final accumulation microstep
- pin CUDA input memory: enabled
- fused optimizer preference: enabled with capability fallback
- native GQA preference: enabled with capability fallback

Invalid combinations fail preflight with a precise error. Examples include a
global optimizer batch not divisible by world size or accumulation geometry,
and a distributed CUDA launch without NCCL availability.

## Failure Handling

- Distributed initialization failure terminates the launch with diagnostics.
- Any rank exception terminates the complete process group.
- Rank zero never writes a ready marker before atomic checkpoint publication.
- Unsupported optional fast paths use tested fallbacks and log the resolved
  path once.
- CUDA out-of-memory does not automatically lower batch size because doing so
  could change optimizer batch semantics. The run stops and remains resumable
  from the last valid checkpoint.
- A two-GPU DDP failure is not silently converted to DataParallel. Single-GPU
  fallback occurs only when one CUDA device is actually available.

## Testing And Acceptance

### Local Tests

- Pure tests for distributed environment detection and rank-aware batch
  geometry.
- Deterministic tests proving rank shards are disjoint and reconstruct the
  original stream order.
- Tests proving `no_sync()` is used only for non-final accumulation steps.
- Tests proving only rank zero owns checkpoint and log side effects.
- Checkpoint compatibility tests across plain, DataParallel, and DDP wrappers.
- Fused AdamW, pinned transfer, label reuse, and native GQA routing tests with
  explicit fallback coverage.
- CPU/Gloo subprocess smoke test when local PyTorch distributed support is
  available.
- Full unit suite, quality scan, and Python compilation.

### Kaggle Gate

Before continuing the long run, resume the same valid checkpoint for a short
20-to-50-step validation:

1. Confirm both GPUs are active under separate ranks.
2. Confirm global batch, accumulation, sequence length, and cumulative tokens
   match the existing profile.
3. Confirm loss and gradient norms remain finite without new spike warnings.
4. Publish and reload one checkpoint.
5. Compare optimizer-step time against the previous DataParallel log.

The DDP path becomes the recommended Kaggle mode only after this gate passes.
No fixed speed multiplier is guaranteed because Kaggle image, storage, and GPU
contention vary.

## Recovery And GitHub Continuity

The implementation is committed and pushed to GitHub only after verification.
On Kaggle, the user can stop the current process, pull `main`, locate the latest
checkpoint that passes the checkpoint contract, and resume with the same
`RESUME_FROM` interface. No checkpoint conversion command is required.
