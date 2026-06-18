# Lightning H100 Quality-Fast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume LaflaGPT Mini 100M pretraining from the verified step-019500 artifact on one H100 80GB while preserving a 32-sequence optimizer batch across the full context curriculum.

**Architecture:** Extend `TrainingConfig` with a stage-aligned CUDA micro-batch schedule and recompute batch geometry whenever curriculum stage changes. Add an H100 profile plus fail-closed bootstrap/background launch scripts that restore only the pretraining artifact, verify SHA-256 and checkpoint format, and retry only explicit CUDA OOM failures with smaller schedule scales.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA 12.8, YAML configs, Bash, unittest, Lightning Studio, GitHub Releases.

---

### Task 1: Stage-aware batch configuration

**Files:**
- Modify: `src/lafla_ai_core/config/schema.py`
- Modify: `src/lafla_ai_core/training/parallelism.py`
- Modify: `src/lafla_ai_core/cli/train_pretrain.py`
- Modify: `tests/unit/test_config_preflight.py`
- Modify: `tests/unit/test_training_parallelism.py`

- [ ] **Step 1: Write failing schema and geometry tests**

Add tests that parse `(32, 16, 8, 4, 2, 1)`, reject a schedule whose length differs from `sequence_curriculum`, and expect stage 3 to resolve micro-batch `4`, accumulation `8`, optimizer batch `32`.

```python
geometry = resolve_stage_batch_geometry(
    configured_micro_batch_size=1,
    configured_gradient_accumulation_steps=16,
    cuda_micro_batch_size_per_device=32,
    cuda_micro_batch_size_per_device_curriculum=(32, 16, 8, 4, 2, 1),
    target_sequences_per_optimizer_step=32,
    stage_index=3,
    decision=single_cuda_decision,
)
self.assertEqual((geometry.per_process_micro_batch_size, geometry.gradient_accumulation_steps), (4, 8))
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m unittest tests.unit.test_config_preflight tests.unit.test_training_parallelism -v`

Expected: failure because the new field/function does not exist.

- [ ] **Step 3: Implement schema and resolver**

Add `cuda_micro_batch_size_per_device_curriculum: tuple[int, ...] = ()` and `cuda_batch_scale: float = 1.0`, parse them, and validate positivity, stage count, CUDA tuning requirements, and `0 < cuda_batch_scale <= 1`. Add:

```python
def resolve_stage_batch_geometry(*, stage_index: int, cuda_micro_batch_size_per_device_curriculum: tuple[int, ...], **kwargs) -> BatchGeometry:
    selected = kwargs["cuda_micro_batch_size_per_device"]
    if cuda_micro_batch_size_per_device_curriculum:
        if not 0 <= stage_index < len(cuda_micro_batch_size_per_device_curriculum):
            raise ValueError("curriculum stage batch programi disinda")
        selected = cuda_micro_batch_size_per_device_curriculum[stage_index]
return resolve_batch_geometry(cuda_micro_batch_size_per_device=selected, **kwargs)
```

Add a pure `scale_cuda_micro_batch_program` helper that accepts only `1.0`, `0.5`, or `0.25`, scales both static and curriculum values to positive divisors of the target optimizer batch, and returns a replaced `TrainingConfig`. Expose it through `train_pretrain --cuda-batch-scale`; validate the base config, apply the scale, then validate the resolved config again.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `python -m unittest tests.unit.test_config_preflight tests.unit.test_training_parallelism -v`

- [ ] **Step 5: Commit**

```bash
git add src/lafla_ai_core/config/schema.py src/lafla_ai_core/training/parallelism.py src/lafla_ai_core/cli/train_pretrain.py tests/unit/test_config_preflight.py tests/unit/test_training_parallelism.py
git commit -m "Add curriculum-aware CUDA batch geometry"
```

### Task 2: Recompute geometry during resume and curriculum transitions

**Files:**
- Modify: `src/lafla_ai_core/training/runner.py`
- Modify: `tests/unit/test_training_runner_smoke.py`
- Modify: `tests/unit/test_training_curriculum.py`

- [ ] **Step 1: Write failing transition tests**

Add a helper-level test proving stage 0 resolves `32/1` and stage 1 resolves `16/2`. Add a smoke runner assertion that the health record includes the stage-selected micro-batch, `cuda_batch_scale`, and constant `sequences_per_optimizer_step: 32`.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m unittest tests.unit.test_training_curriculum tests.unit.test_training_runner_smoke -v`

- [ ] **Step 3: Implement active geometry refresh**

Create `_resolve_active_batch_geometry(training_config, parallel_decision, stage)` and call it after resume stage selection and inside `if stage.index != active_stage.index`. Update all four active values before rebuilding the iterator:

```python
batch_geometry = _resolve_active_batch_geometry(training_config, parallel_decision, active_stage)
active_process_micro_batch_size = batch_geometry.per_process_micro_batch_size
active_global_micro_batch_size = batch_geometry.global_micro_batch_size
active_gradient_accumulation_steps = batch_geometry.gradient_accumulation_steps
```

Keep optimizer/model objects unchanged and log the refreshed values.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `python -m unittest tests.unit.test_training_curriculum tests.unit.test_training_runner_smoke -v`

- [ ] **Step 5: Commit**

```bash
git add src/lafla_ai_core/training/runner.py tests/unit/test_training_runner_smoke.py tests/unit/test_training_curriculum.py
git commit -m "Refresh batch geometry at curriculum transitions"
```

### Task 3: Add the H100 quality-fast profile

**Files:**
- Create: `configs/training/lightning/lightning-h100-100m-quality-fast.yaml`
- Modify: `tests/unit/test_config_preflight.py`

- [ ] **Step 1: Write a failing profile contract test**

Assert BF16, fused optimizer preference, 6B target tokens, optimizer batch 32, the exact six-stage schedule, 10M token checkpoints, eight retained checkpoints, and at least 20GB free-space guard.

- [ ] **Step 2: Run test and confirm RED**

Run: `python -m unittest tests.unit.test_config_preflight -v`

- [ ] **Step 3: Create the profile**

Use the approved curriculum and:

```yaml
precision: bf16
prefer_fused_optimizer: true
cuda_micro_batch_size_per_device: 32
cuda_micro_batch_size_per_device_curriculum: [32, 16, 8, 4, 2, 1]
target_sequences_per_optimizer_step: 32
checkpoint_every_tokens: 10000000
keep_last_checkpoints: 8
checkpoint_min_free_gb: 20.0
```

Do not change LR, warmup, curriculum boundaries, or target tokens from the stable profile.

- [ ] **Step 4: Run test and confirm GREEN**

Run: `python -m unittest tests.unit.test_config_preflight -v`

- [ ] **Step 5: Commit**

```bash
git add configs/training/lightning/lightning-h100-100m-quality-fast.yaml tests/unit/test_config_preflight.py
git commit -m "Add H100 quality-fast training profile"
```

### Task 4: Generalize launcher metadata and add H100 launchers

**Files:**
- Modify: `scripts/lightning/start_t4_100m.sh`
- Modify: `scripts/lightning/monitor_100m.sh`
- Create: `scripts/lightning/start_h100_100m.sh`
- Create: `scripts/lightning/start_h100_100m_background.sh`
- Create: `scripts/lightning/monitor_h100_100m.sh`
- Create: `tests/unit/test_lightning_h100_launcher.py`

- [ ] **Step 1: Write failing launcher text-contract tests**

Require the H100 wrapper to select the H100 config, `torch==2.11.0` from `cu128`, an H100-specific venv/log, BF16/H100 runtime checks, background PID protection, monitor log override, and ordered `1.0 0.5 0.25` OOM retries.

- [ ] **Step 2: Run test and confirm RED**

Run: `python -m unittest tests.unit.test_lightning_t4_launcher tests.unit.test_lightning_h100_launcher -v`

- [ ] **Step 3: Parameterize the shared launcher**

Expose environment defaults for `PROFILE_NAME`, `VENV`, `LOG`, `DATASET_VERSION`, `ARCHIVE_NAME`, `PYTORCH_INDEX_URL`, `PYTORCH_VERSION`, and `DEVICE_LABEL`; preserve current T4 defaults exactly.

- [ ] **Step 4: Add H100 wrappers**

`start_h100_100m.sh` sets H100 defaults and calls the shared launcher with `--cuda-batch-scale` for scales `1.0`, `0.5`, and `0.25`. It advances only when the failed log contains `torch.OutOfMemoryError` or `CUDA out of memory`; every other non-zero exit fails immediately. Each retry relies on the shared launcher's latest-`READY.json` auto-resume. The background wrapper uses `nohup`, refuses a second `train_pretrain` process, and records PID/log paths. The monitor wrapper exports the H100 nohup log before executing the common monitor.

- [ ] **Step 5: Run tests and confirm GREEN**

Run: `python -m unittest tests.unit.test_lightning_t4_launcher tests.unit.test_lightning_h100_launcher -v`

- [ ] **Step 6: Commit**

```bash
git add scripts/lightning tests/unit/test_lightning_t4_launcher.py tests/unit/test_lightning_h100_launcher.py
git commit -m "Add H100 background training launcher"
```

### Task 5: Add fail-closed GitHub Release restore

**Files:**
- Create: `scripts/lightning/bootstrap_h100_100m.sh`
- Modify: `tests/unit/test_lightning_h100_launcher.py`

- [ ] **Step 1: Write failing restore contract tests**

Require exact Release asset names, `sha256sum -c`, Zstandard integrity checks, temporary extraction, `lafla-trainer-state-v2` validation, rejection of `lafla-thinking-sft-state-v1`, and no data preparation fallback during restore.

- [ ] **Step 2: Run test and confirm RED**

Run: `python -m unittest tests.unit.test_lightning_h100_launcher -v`

- [ ] **Step 3: Implement bootstrap**

Download `SHA256SUMS`, `pretrain-lafla-step-019500.tar.zst`, and `runtime-metadata.tar.zst` from tag `lafla-100m-backup-20260618`. Verify only the required manifest lines, run `zstd -t`, extract under a temporary directory, validate JSON format and required files, then move into `$WORK` without overwriting an existing valid workspace.

- [ ] **Step 4: Run test and confirm GREEN**

Run: `python -m unittest tests.unit.test_lightning_h100_launcher -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/lightning/bootstrap_h100_100m.sh tests/unit/test_lightning_h100_launcher.py
git commit -m "Add verified H100 checkpoint restore"
```

### Task 6: Full verification, publish, restore, and start

**Files:**
- Verify all changed files.
- Remote targets: `/teamspace/studios/this_studio/LaflaAi-Core`, `/teamspace/studios/this_studio/LaflaAI100M`

- [ ] **Step 1: Run focused and full local verification**

Run:

```bash
PYTHONPATH=src python -m unittest tests.unit.test_config_preflight tests.unit.test_training_parallelism tests.unit.test_training_curriculum tests.unit.test_training_runner_smoke tests.unit.test_lightning_t4_launcher tests.unit.test_lightning_h100_launcher -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python -m lafla_ai_core.cli.quality_scan --root .
```

Expected: zero failures and quality scan `ok: true`.

- [ ] **Step 2: Push `main`**

Run: `git push origin main` and verify `git ls-remote origin refs/heads/main` equals local HEAD.

- [ ] **Step 3: Bootstrap the H100 host**

Clone `main`, run `bash scripts/lightning/bootstrap_h100_100m.sh`, and verify checkpoint/tokenizer/data files plus SHA checks.

- [ ] **Step 4: Start background training**

Run: `bash scripts/lightning/start_h100_100m_background.sh`.

- [ ] **Step 5: Verify live training**

Confirm one training process, H100 utilization, BF16/fused AdamW/native GQA health fields, resume from step 019500, finite loss, optimizer batch 32, and creation of new checkpoint/snapshot paths.
