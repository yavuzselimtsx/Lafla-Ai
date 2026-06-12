# Kaggle Safe Speed Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate LaflaGPT Mini 100M Kaggle T4 x2 pretraining while preserving model/checkpoint compatibility and the optimizer-step token budget.

**Architecture:** Add independently tested resolution helpers for CUDA batch geometry and runtime activation checkpointing. Use CUDA SDPA when local attention covers the complete active sequence, while retaining the current bounded implementation for longer contexts.

**Tech Stack:** Python, PyTorch, `unittest`, YAML configuration, Kaggle T4 x2.

---

### Task 1: Resolve Safe Batch Geometry

**Files:**
- Modify: `src/lafla_ai_core/config/schema.py`
- Modify: `src/lafla_ai_core/training/parallelism.py`
- Modify: `tests/unit/test_training_parallelism.py`

- [ ] Add failing tests proving that T4 x2 resolves to global micro batch 4 and accumulation 8, while one CUDA device resolves to micro batch 2 and accumulation 16.
- [ ] Run `PYTHONPATH=src python -m unittest tests.unit.test_training_parallelism -v` and verify the new tests fail.
- [ ] Add optional `cuda_micro_batch_size_per_device` and `target_sequences_per_optimizer_step` fields with zero meaning disabled.
- [ ] Implement a pure `resolve_batch_geometry` helper that preserves existing behavior when the fields are disabled.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Add Runtime Checkpointing Policy

**Files:**
- Modify: `src/lafla_ai_core/config/schema.py`
- Modify: `src/lafla_ai_core/model/transformer.py`
- Modify: `src/lafla_ai_core/training/runner.py`
- Modify: `tests/unit/test_model_transformer.py`
- Modify: `tests/unit/test_training_parallelism.py`

- [ ] Add failing tests proving checkpointing can be disabled at runtime and that the threshold policy returns false for 2048/4096 and true for 8192.
- [ ] Run the focused tests and verify the failures describe missing runtime policy.
- [ ] Add `gradient_checkpointing_min_sequence_length`, defaulting to zero for backward-compatible behavior.
- [ ] Add a model runtime toggle and update it whenever the curriculum stage changes.
- [ ] Log the resolved checkpointing state and re-run focused tests.

### Task 3: Use SDPA For Full-Window Local Attention

**Files:**
- Modify: `src/lafla_ai_core/model/transformer.py`
- Modify: `tests/unit/test_model_transformer.py`

- [ ] Add a failing test that patches `scaled_dot_product_attention` and proves it is selected when sequence length does not exceed the sliding window.
- [ ] Add a second test proving longer sequences retain the bounded local path.
- [ ] Implement the minimal SDPA fast path with the same causal and dropout settings as global attention.
- [ ] Run transformer tests and verify both routing and numerical tests pass.

### Task 4: Wire The Kaggle Profile

**Files:**
- Modify: `configs/training/kaggle/kaggle-gpu-100m.yaml`
- Modify: `docs/operations/kaggle-gpu-100m.md`
- Modify: `tests/unit/test_config_preflight.py`

- [ ] Configure per-device micro batch 2, target optimizer batch 32, and checkpointing threshold 8192.
- [ ] Document expected T4 x2 and single-GPU resolved values and the OOM rollback procedure.
- [ ] Run preflight and focused config tests.

### Task 5: Verify And Publish

**Files:**
- Verify all modified files.

- [ ] Run `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`.
- [ ] Run `PYTHONPATH=src python -m lafla_ai_core.cli.quality_scan --root .`.
- [ ] Run `python -m compileall -q src tests scripts`.
- [ ] Commit the verified change and push it to GitHub `main`.
- [ ] Provide Kaggle checkpoint-stop, pull, and resume commands.

