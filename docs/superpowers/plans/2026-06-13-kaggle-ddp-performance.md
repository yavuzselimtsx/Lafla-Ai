# Kaggle DDP Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kaggle T4 x2 DataParallel hot path with checkpoint-compatible DDP and add capability-gated CUDA fast paths without changing LaflaGPT Mini's model weights, global optimizer batch, token accounting, data sources, or learning-rate schedule.

**Architecture:** A focused distributed-runtime module owns rank discovery, NCCL initialization, barriers, reduction, and cleanup. Torch-independent policy helpers own batch geometry, rank sharding, and gradient-sync decisions; the training runner composes them while rank zero exclusively owns persistent side effects. Optional fused AdamW, pinned transfer, and native GQA are enabled only after capability checks and retain explicit fallbacks.

**Tech Stack:** Python 3.12, PyTorch Distributed/DDP/NCCL, CUDA SDPA, `unittest`, YAML, Bash, Kaggle T4 x2.

---

### Task 1: Add Distributed Configuration And Pure Policies

**Files:**
- Modify: `src/lafla_ai_core/config/schema.py`
- Modify: `src/lafla_ai_core/training/parallelism.py`
- Modify: `tests/unit/test_training_parallelism.py`
- Modify: `tests/unit/test_config_preflight.py`

- [ ] **Step 1: Write failing config and policy tests**

Add tests that require:

```python
config = _training_config(
    data_parallel="auto",
    distributed_backend="auto",
    gradient_sync="final_microstep",
    pin_memory=True,
    prefer_fused_optimizer=True,
    prefer_native_gqa=True,
)
config.validate()

decision = resolve_parallelism(
    data_parallel="auto",
    device_type="cuda",
    cuda_device_count=2,
    distributed_world_size=2,
)
geometry = resolve_batch_geometry(
    configured_micro_batch_size=1,
    configured_gradient_accumulation_steps=16,
    cuda_micro_batch_size_per_device=2,
    target_sequences_per_optimizer_step=32,
    decision=decision,
)
assert geometry.per_process_micro_batch_size == 2
assert geometry.global_micro_batch_size == 4
assert geometry.gradient_accumulation_steps == 8
assert should_sync_gradients(6, 8) is False
assert should_sync_gradients(7, 8) is True
assert tuple(iter_rank_positions(range(8), rank=0, world_size=2)) == (0, 2, 4, 6)
assert tuple(iter_rank_positions(range(8), rank=1, world_size=2)) == (1, 3, 5, 7)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.unit.test_training_parallelism tests.unit.test_config_preflight -v
```

Expected: failures for missing configuration fields, `resolve_parallelism`,
`per_process_micro_batch_size`, `should_sync_gradients`, and
`iter_rank_positions`.

- [ ] **Step 3: Implement minimal typed configuration**

Add backward-compatible `TrainingConfig` fields:

```python
distributed_backend: str = "auto"
gradient_sync: str = "every_microstep"
pin_memory: bool = False
prefer_fused_optimizer: bool = False
prefer_native_gqa: bool = False
```

Parse them in `from_mapping()` and validate:

```python
_require(self.distributed_backend in {"auto", "nccl", "gloo"}, "desteklenmeyen distributed_backend")
_require(self.gradient_sync in {"every_microstep", "final_microstep"}, "desteklenmeyen gradient_sync")
```

Keep defaults equivalent to the pre-DDP behavior for all existing profiles.

- [ ] **Step 4: Implement pure parallelism policies**

Extend `ParallelismDecision` with `world_size`, `rank`, and `local_rank`, then
implement:

```python
def resolve_parallelism(
    data_parallel: str,
    device_type: str,
    cuda_device_count: int,
    distributed_world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
) -> ParallelismDecision:
    ...

def should_sync_gradients(micro_step: int, accumulation_steps: int) -> bool:
    if not 0 <= micro_step < accumulation_steps:
        raise ValueError("micro_step accumulation araliginda olmali")
    return micro_step == accumulation_steps - 1

def iter_rank_positions(source, *, rank: int, world_size: int):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size hatali")
    for index, value in enumerate(source):
        if index % world_size == rank:
            yield value
```

`BatchGeometry` must expose both per-process and global micro batch. Existing
single-device and legacy DataParallel tests must retain their previous global
batch results.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all focused tests pass, with
Torch-dependent tests skipped only when Torch is not installed.

- [ ] **Step 6: Commit**

```bash
git add src/lafla_ai_core/config/schema.py src/lafla_ai_core/training/parallelism.py tests/unit/test_training_parallelism.py tests/unit/test_config_preflight.py
git commit -m "feat: define safe distributed training policies"
```

### Task 2: Implement Distributed Runtime Lifecycle

**Files:**
- Create: `src/lafla_ai_core/training/distributed.py`
- Create: `tests/unit/test_training_distributed.py`

- [ ] **Step 1: Write failing lifecycle tests**

Use patched environment and Torch distributed functions to require:

```python
environment = read_distributed_environment({"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
assert environment.world_size == 2
assert environment.rank == 1
assert environment.local_rank == 1
assert environment.is_distributed
assert not environment.is_primary
```

Also require that malformed/incomplete rank variables fail closed, backend
`auto` resolves CUDA to `nccl`, and single-process execution never initializes a
process group.

- [ ] **Step 2: Run focused test and verify RED**

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.unit.test_training_distributed -v
```

Expected: import failure because `training.distributed` does not exist.

- [ ] **Step 3: Implement the lifecycle module**

Create immutable `DistributedEnvironment` and `DistributedRuntime` types.
`initialize_distributed()` must:

```python
environment = read_distributed_environment(os.environ)
if environment.is_distributed:
    if device_type != "cuda":
        raise RuntimeError("Kaggle DDP yolu CUDA gerektirir")
    backend = "nccl" if configured_backend == "auto" else configured_backend
    if backend == "nccl" and not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL kullanilabilir degil")
    torch.cuda.set_device(environment.local_rank)
    torch.distributed.init_process_group(backend=backend, init_method="env://")
```

Expose `barrier()`, `mean_tensor()`, `is_primary`, and idempotent `close()`.
`mean_tensor()` clones/detaches its input, all-reduces with SUM, and divides by
world size only in distributed mode.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all lifecycle tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lafla_ai_core/training/distributed.py tests/unit/test_training_distributed.py
git commit -m "feat: add fail-closed ddp runtime lifecycle"
```

### Task 3: Integrate DDP, Rank Sharding, And Final-Microstep Sync

**Files:**
- Modify: `src/lafla_ai_core/training/runner.py`
- Modify: `src/lafla_ai_core/cli/train_pretrain.py`
- Modify: `tests/unit/test_training_parallelism.py`
- Modify: `tests/unit/test_training_runner_smoke.py`
- Create: `tests/unit/test_training_rank_sharding.py`

- [ ] **Step 1: Write failing data-shard and synchronization tests**

Require `JsonlTokenBlockDataset` and `SmokeTokenBlockDataset` to accept
`rank/world_size`; two rank iterators must be disjoint and their interleaving
must reconstruct the unsharded deterministic smoke stream.

Use a recording module with `no_sync()` to require:

```python
contexts = [
    gradient_sync_context(model, micro_step=index, accumulation_steps=8, final_microstep_only=True)
    for index in range(8)
]
assert model.no_sync_calls == 7
```

Require the CLI to print its JSON summary only when the runtime is primary.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.unit.test_training_rank_sharding tests.unit.test_training_parallelism tests.unit.test_training_runner_smoke -v
```

Expected: failures for missing rank arguments and gradient-sync helper.

- [ ] **Step 3: Add deterministic rank sharding**

Shard the complete deterministic block iterator:

```python
blocks = iter_packed_token_blocks(...)
yield from iter_rank_positions(blocks, rank=self.rank, world_size=self.world_size)
```

For smoke data, generate the same seeded global sequence on every rank and apply
the same positional sharding. Do not add rank to the random seed.

- [ ] **Step 4: Wire DDP and per-process batches into the runner**

Initialize the distributed runtime before device selection. On DDP:

```python
device = torch.device("cuda", runtime.local_rank)
train_model = torch.nn.parallel.DistributedDataParallel(
    _TrainingLossModule(base_model),
    device_ids=[runtime.local_rank],
    output_device=runtime.local_rank,
)
```

Build each DataLoader with `geometry.per_process_micro_batch_size`. Continue to
pass `geometry.global_micro_batch_size` to `tokens_per_optimizer_step()`.

For accumulation:

```python
with gradient_sync_context(
    train_model,
    micro_step=micro_step,
    accumulation_steps=active_gradient_accumulation_steps,
    final_microstep_only=training_config.gradient_sync == "final_microstep",
):
    with _autocast_context(device, dtype):
        raw_loss = train_model(batch, labels=batch)
        ...
    scaler.scale(loss).backward()
```

Reduce the detached accumulated loss for rank-zero health reporting. Gradient
clipping remains after final synchronization.

- [ ] **Step 5: Restrict persistent side effects to rank zero**

Only primary rank creates/writes health logs, checkpoints, retention changes,
and final artifacts. Wrap periodic and final saves with pre/post barriers.
Preserve the existing unwrapped `base_model` checkpoint payload and
`RESUME_FROM` loading path.

All ranks close the process group in `finally`. The CLI uses the runtime/summary
primary flag to avoid duplicate JSON output.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2, plus:

```powershell
python -m unittest tests.unit.test_checkpoint_publish tests.unit.test_checkpoint_contract -v
```

Expected: focused tests pass and checkpoint-contract tests remain green.

- [ ] **Step 7: Commit**

```bash
git add src/lafla_ai_core/training/runner.py src/lafla_ai_core/cli/train_pretrain.py tests/unit/test_training_parallelism.py tests/unit/test_training_runner_smoke.py tests/unit/test_training_rank_sharding.py
git commit -m "feat: train with rank-sharded ddp accumulation"
```

### Task 4: Add Capability-Gated CUDA Fast Paths

**Files:**
- Modify: `src/lafla_ai_core/training/runner.py`
- Modify: `src/lafla_ai_core/model/transformer.py`
- Modify: `tests/unit/test_training_parallelism.py`
- Modify: `tests/unit/test_model_transformer.py`

- [ ] **Step 1: Write failing optimizer, transfer, and GQA tests**

Require:

```python
optimizer, mode = build_optimizer(model, config, device=torch.device("cuda"))
assert mode in {"fused_adamw", "adamw"}
```

Patch AdamW to reject `fused=True` and assert the builder retries without it.
Require the training iterator to pass `pin_memory=True` only for configured
CUDA runs and require batch transfer to use `non_blocking=True`.

For GQA, patch SDPA and assert `enable_gqa=True` receives unexpanded KV heads
when a capability probe succeeds; when it fails, assert explicit
`repeat_interleave` produces the previous head count.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.unit.test_training_parallelism tests.unit.test_model_transformer -v
```

Expected: failures for missing optimizer mode/capability and native-GQA routing.

- [ ] **Step 3: Implement fused AdamW preference**

Change `build_optimizer()` to return an optimizer plus a resolved mode. Request
`fused=True` only for AdamW, configured CUDA, and a supporting constructor.
Catch only constructor capability errors (`TypeError` or a clear unsupported
`RuntimeError`), then retry the identical parameter groups and hyperparameters
without `fused`.

The optimizer state dict must remain the standard AdamW state dict.

- [ ] **Step 4: Implement pinned non-blocking transfer and label reuse**

Pass `pin_memory=training_config.pin_memory and device.type == "cuda"` to
DataLoader. Transfer with:

```python
batch = next(iterator).to(
    device,
    non_blocking=training_config.pin_memory and device.type == "cuda",
)
labels = batch
```

The model loss path must not mutate `labels`.

- [ ] **Step 5: Implement native GQA capability routing**

Add a runtime model toggle and a small CUDA probe that calls SDPA with query
heads divisible by KV heads and `enable_gqa=True`. Enable native GQA only when
the probe succeeds. Otherwise retain the exact explicit expansion path.

Local long-window manual attention and XLA always retain explicit KV expansion.
Global CUDA and full-window local CUDA may use native GQA.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass or Torch-only tests are
explicitly skipped when Torch is unavailable.

- [ ] **Step 7: Commit**

```bash
git add src/lafla_ai_core/training/runner.py src/lafla_ai_core/model/transformer.py tests/unit/test_training_parallelism.py tests/unit/test_model_transformer.py
git commit -m "perf: add capability-gated cuda training paths"
```

### Task 5: Launch With Torchrun And Document Recovery

**Files:**
- Modify: `scripts/kaggle/start_gpu_100m.sh`
- Modify: `configs/training/kaggle/kaggle-gpu-100m.yaml`
- Modify: `docs/operations/kaggle-gpu-100m.md`
- Modify: `src/lafla_ai_core/kaggle/run_plan.py`
- Modify: `tests/unit/test_kaggle_run_plan.py`
- Modify: `tests/unit/test_config_preflight.py`

- [ ] **Step 1: Write failing launcher/config tests**

Require the Kaggle profile to resolve:

```python
assert training.distributed_backend == "auto"
assert training.gradient_sync == "final_microstep"
assert training.pin_memory
assert training.prefer_fused_optimizer
assert training.prefer_native_gqa
```

Require generated Kaggle commands and launcher text to include `torchrun` for
multi-GPU execution while retaining the same CLI arguments and `RESUME_FROM`.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.unit.test_kaggle_run_plan tests.unit.test_config_preflight -v
```

Expected: failures for missing DDP profile fields and launcher selection.

- [ ] **Step 3: Configure the safe DDP profile**

Add:

```yaml
distributed_backend: auto
gradient_sync: final_microstep
pin_memory: true
prefer_fused_optimizer: true
prefer_native_gqa: true
```

Keep the current micro batch, target sequences, learning rate, curriculum,
target tokens, and checkpoint cadence unchanged.

- [ ] **Step 4: Select torchrun in the Kaggle launcher**

After dependency installation, detect CUDA count through Python. Use:

```bash
if [ "$CUDA_DEVICE_COUNT" -ge 2 ]; then
  TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node "$CUDA_DEVICE_COUNT")
else
  TRAIN_LAUNCHER=(python)
fi

TRAIN_ARGS=(-m lafla_ai_core.cli.train_pretrain ...)
"${TRAIN_LAUNCHER[@]}" "${TRAIN_ARGS[@]}"
```

Do not launch DataParallel as an automatic fallback after a DDP failure.

- [ ] **Step 5: Update operator documentation**

Document rank-local batch `2`, global micro batch `4`, accumulation `8`,
65,536 tokens per optimizer step, latest-valid-checkpoint resume commands, and
the 20-to-50-step Kaggle acceptance gate.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2 and:

```powershell
bash -n scripts/kaggle/start_gpu_100m.sh
```

Expected: tests pass and shell syntax is valid where Bash is available.

- [ ] **Step 7: Commit**

```bash
git add scripts/kaggle/start_gpu_100m.sh configs/training/kaggle/kaggle-gpu-100m.yaml docs/operations/kaggle-gpu-100m.md src/lafla_ai_core/kaggle/run_plan.py tests/unit/test_kaggle_run_plan.py tests/unit/test_config_preflight.py
git commit -m "ops: launch kaggle multi-gpu training with torchrun"
```

### Task 6: Full Verification And Publication

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run the complete unit suite**

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: zero failures. Torch-dependent skips are reported explicitly.

- [ ] **Step 2: Run static quality and compilation gates**

```powershell
python -m lafla_ai_core.cli.quality_scan --root .
python -m compileall -q src tests scripts
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Verify checkpoint and batch invariants**

Run focused tests proving:

```text
per-rank batch = 2
world size = 2
global micro batch = 4
accumulation = 8
sequences per optimizer step = 32
tokens per 2048-token optimizer step = 65,536
checkpoint state keys have no module. prefix
```

- [ ] **Step 4: Review the complete implementation**

Inspect the diff against
`docs/superpowers/specs/2026-06-13-kaggle-ddp-performance-design.md`. Reject
unrequested architecture changes, hidden batch changes, broad exception
fallbacks, rank-unsafe file writes, and hardcoded GPU product names.

- [ ] **Step 5: Merge and push**

After all gates pass, merge the isolated feature branch into `main`, push
GitHub, and verify local `main` matches `origin/main`.

- [ ] **Step 6: Provide exact Kaggle continuation commands**

Give commands to stop only the active training process, `git pull --ff-only`,
validate/select the latest checkpoint, run a short DDP acceptance test, and
resume the long run using the existing `RESUME_FROM` interface.
