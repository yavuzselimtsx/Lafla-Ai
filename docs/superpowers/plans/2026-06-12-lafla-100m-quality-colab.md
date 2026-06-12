# Lafla 100M Quality Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert LaflaAi-Core from hardcoded 380M/400M flows into a configurable 98M Turkish/German-first model family with 20K context, bounded conversation memory, Transformers KV cache, Colab TPU curriculum, measured CPU memory, and a future 200M depth-growth path.

**Architecture:** Typed configs remain the source of truth. The 100M model uses 12x768 SwiGLU blocks, 12 query heads, 2 KV heads, and a configurable local/global attention pattern. Runtime conversation assembly, summarization, retrieval, memory estimation, checkpoint growth, and Colab orchestration live in separate focused modules.

**Tech Stack:** Python 3.11+, PyTorch/PyTorch-XLA, Hugging Face Transformers/tokenizers, YAML/JSON configs, unittest, psutil for Windows RSS measurement.

---

### Task 1: Repair atomic checkpoints and establish 100M typed configs

**Files:**
- Modify: `src/lafla_ai_core/model/checkpoint_io.py`
- Modify: `src/lafla_ai_core/config/schema.py`
- Modify: `src/lafla_ai_core/cli/preflight.py`
- Create: `configs/model/lafla-100m-thinking.yaml`
- Create: `configs/tokenizer/turkish-german-thinking-bpe.yaml`
- Create: `configs/runtime/desktop-i3-int8-100m.yaml`
- Create: `configs/training/colab/colab-tpu-v5e-100m.yaml`
- Test: `tests/unit/test_checkpoint_publish.py`
- Test: `tests/unit/test_config_preflight.py`
- Test: `tests/unit/test_model_size.py`

- [ ] Write a failing filesystem test proving `_publish_directory()` replaces a completed target and removes its backup.
- [ ] Run the checkpoint test and confirm the current implementation fails because the target is not published.
- [ ] Move the unreachable atomic replacement block back into `_publish_directory()`.
- [ ] Add `attention_pattern`, `sliding_window`, and `rope_scaling` to `ModelConfig`.
- [ ] Add runtime cache, token-budget, quantization, RSS, and concurrency fields to `RuntimeConfig`.
- [ ] Add token-based curriculum fields to `TrainingConfig`.
- [ ] Add cross-config checks for runtime/model context and tokenizer/model vocabulary.
- [ ] Add the four 100M configs and assert the parameter estimate is between 95M and 105M.
- [ ] Run focused config/checkpoint/model-size tests.

### Task 2: Implement configurable hybrid attention and long-context position handling

**Files:**
- Modify: `src/lafla_ai_core/model/transformer.py`
- Modify: `src/lafla_ai_core/model/size.py`
- Test: `tests/unit/test_model_transformer.py`
- Test: `tests/unit/test_model_attention_pattern.py`

- [ ] Write failing pure-config tests for the 3-global/9-local layer pattern.
- [ ] Write Torch tests for local causal visibility, global visibility, and shape/loss behavior.
- [ ] Pass layer index and resolved attention mode into each decoder block.
- [ ] Apply configurable sliding-window masks on CPU/CUDA and XLA FlashAttention when available.
- [ ] Keep a fail-closed XLA fallback that rejects unsafe 20K quadratic local attention instead of silently OOMing.
- [ ] Add configurable RoPE scaling with stable defaults.
- [ ] Run focused model tests; record Torch-dependent skips if Torch is unavailable.

### Task 3: Make the Hugging Face package cache-aware and model-config driven

**Files:**
- Modify: `src/lafla_ai_core/export/hf_remote_code.py`
- Modify: `src/lafla_ai_core/export/hf_package.py`
- Test: `tests/unit/test_hf_package.py`
- Test: `tests/unit/test_hf_remote_code_contract.py`

- [ ] Write failing tests asserting exported config contains attention pattern, sliding window, RoPE scaling, `use_cache=true`, and no 400M defaults.
- [ ] Replace hardcoded remote config defaults with neutral values overridden by `config.json`.
- [ ] Add `past_key_values`, `cache_position`, attention masks, and cache returns to exported model code.
- [ ] Ensure generation processes only uncached token ids after prefill.
- [ ] Add cache implementation metadata to `generation_config.json`.
- [ ] Run HF package contract tests.

### Task 4: Add token-budgeted conversation summarization and message retrieval contracts

**Files:**
- Create: `src/lafla_ai_core/runtime/conversation_memory.py`
- Create: `src/lafla_ai_core/runtime/message_search.py`
- Create: `src/lafla_ai_core/runtime/prompt_budget.py`
- Test: `tests/unit/test_conversation_memory.py`
- Test: `tests/unit/test_message_search.py`
- Test: `tests/unit/test_prompt_budget.py`

- [ ] Write failing tests for 20,480 total tokens, 15,360 summarization trigger, 4,096 recent-token preservation, and 2,048 retrieval cap.
- [ ] Define typed messages, structured summary sections, source message ids, and transaction results.
- [ ] Implement fail-closed summary replacement: invalid summaries leave source messages and prior summary unchanged.
- [ ] Define an authorization-scoped `search_messages` protocol that accepts no raw SQL/shell input.
- [ ] Rank and pack retrieval results without splitting records mid-message.
- [ ] Assemble final prompt budgets so system, summary, retrieval, recent messages, and output reservation share one total limit.
- [ ] Run focused runtime memory/retrieval tests.

### Task 5: Add derived memory budgets and real process-tree RSS benchmarking

**Files:**
- Create: `src/lafla_ai_core/runtime/memory_budget.py`
- Create: `src/lafla_ai_core/runtime/rss.py`
- Create: `src/lafla_ai_core/cli/benchmark_inference.py`
- Modify: `requirements/colab.txt`
- Create: `requirements/runtime-cpu.txt`
- Test: `tests/unit/test_memory_budget.py`
- Test: `tests/unit/test_rss.py`

- [ ] Write failing tests for weight bytes, hybrid KV bytes, and component totals from model/runtime configs.
- [ ] Implement formulas using parameter count, dtype bytes, local/global cache lengths, batch size, and configured overhead.
- [ ] Implement Windows-compatible process-tree peak RSS/USS sampling with psutil.
- [ ] Add a CLI that runs 2K/8K/15K/20K profiles and writes JSON without claiming success when the model/dependencies are missing.
- [ ] Fail the release gate when measured peak RSS exceeds 700 MiB.
- [ ] Run focused memory and RSS tests.

### Task 6: Add 100M real-data policy, uncertainty contracts, and release gates

**Files:**
- Create: `configs/data/source-plans/lafla-100m-source-plan.json`
- Create: `configs/data/identity/lafla-model-identity-100m.jsonl`
- Modify: `configs/data/lafla-data-policy.yaml`
- Modify: `configs/evaluation/release-gates.yaml`
- Create: `src/lafla_ai_core/evaluation/uncertainty.py`
- Create: `src/lafla_ai_core/evaluation/long_context.py`
- Test: `tests/unit/test_data_source_plan.py`
- Test: `tests/unit/test_uncertainty_gates.py`
- Test: `tests/unit/test_long_context_gates.py`

- [ ] Write failing tests for Turkish/German domain coverage, weights summing to one, and review-required sources staying non-primary.
- [ ] Add the 42/23/10/10/8/7 source-domain plan with real source identifiers and explicit review states.
- [ ] Add paired answerable/unanswerable, contradictory evidence, stale-current-fact, and false-premise evaluation contracts.
- [ ] Add deterministic passkey/needle result scoring independent of model execution.
- [ ] Require abstention, source faithfulness, and long-context gates in release config.
- [ ] Run focused data/evaluation tests.

### Task 7: Generalize Colab TPU orchestration and add staged 100M curriculum

**Files:**
- Modify: `src/lafla_ai_core/colab/run_plan.py`
- Modify: `src/lafla_ai_core/cli/colab_plan.py`
- Create: `scripts/colab/start_tpu_v5e_100m.sh`
- Create: `notebooks/colab/lafla_tpu_100m_training.ipynb`
- Modify: `src/lafla_ai_core/training/runner.py`
- Test: `tests/unit/test_colab_run_plan.py`
- Test: `tests/unit/test_training_curriculum.py`

- [ ] Write failing tests proving Colab paths/configs/model names/artifact names are caller supplied rather than 380M constants.
- [ ] Add typed Colab profile paths for model, training, tokenizer, runtime, identity data, and model name.
- [ ] Generate preflight, data audit, tokenizer, HF package, training, resume, artifact, and archive commands from that profile.
- [ ] Add staged sequence curriculum resolution for 2K, 4K, 8K, 12K, 16K, and 20K.
- [ ] Log cumulative tokens and curriculum stage in trainer state/health logs.
- [ ] Create a real-data-only TPU launcher and notebook that refuse missing manifest/data.
- [ ] Run focused Colab/curriculum tests.

### Task 8: Add function-preserving 100M-to-200M depth growth

**Files:**
- Create: `src/lafla_ai_core/model/growth.py`
- Create: `src/lafla_ai_core/cli/grow_checkpoint.py`
- Create: `configs/model/lafla-200m-thinking.yaml`
- Test: `tests/unit/test_model_growth.py`

- [ ] Write failing tests for compatible family dimensions and deterministic source-to-target layer mapping.
- [ ] Reject tokenizer, hidden-size, head, KV-head, norm, and attention-pattern incompatibilities.
- [ ] Build a depth mapping that preserves all source layers and initializes inserted residual blocks as identity-compatible blocks.
- [ ] Write a conversion report with source/target parameters, layer map, and required continued-pretraining warning.
- [ ] Add a Torch checkpoint conversion CLI; skip execution tests when Torch is unavailable.
- [ ] Run focused growth tests.

### Task 9: Remove 380M defaults from the active 100M path and verify the whole repository

**Files:**
- Modify: `src/lafla_ai_core/training/phase_plan.py`
- Modify: `src/lafla_ai_core/cli/training_phase_plan.py`
- Modify: `README.md`
- Modify: `docs/operations/next-training-plan.md`
- Modify: `docs/operations/low-power-runtime.md`
- Test: `tests/unit/test_training_phase_plan.py`
- Test: `tests/unit/test_static_quality_scan.py`

- [ ] Add a configurable 100M phase plan without deleting legacy 380M/400M profiles.
- [ ] Make README commands point to the 100M Colab TPU workflow as the primary path.
- [ ] Document the CLI's 2K retrieval cap and 15K summary trigger.
- [ ] Add static scan rules for hardcoded active-model paths and `use_cache=false` in HF exports.
- [ ] Run `quality_scan`.
- [ ] Run the full unittest suite.
- [ ] Run all 100M preflight combinations.
- [ ] Report Torch-dependent tests as unverified unless run in Colab/Torch environment.

