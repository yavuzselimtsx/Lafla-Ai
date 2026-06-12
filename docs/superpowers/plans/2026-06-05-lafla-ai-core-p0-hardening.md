# LaflaAi-Core P0 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LaflaAi-Core runtime, tokenizer/chat format, checkpoint, artifact, and HF export contracts become testable and fail-closed before another costly Colab run.

**Architecture:** Keep the current modular boundaries. Add small contract modules instead of growing runner/runtime files. Training remains separate from runtime and export; chat formatting becomes one shared tokenizer contract used by packing, SFT, and Hugging Face metadata.

**Tech Stack:** Python stdlib, `unittest`, existing `tokenizers` integration, existing Torch checkpoint code when Torch is installed.

---

### Task 1: Runtime Output Cleanup

**Files:**
- Modify: `tests/unit/test_runtime_policy.py`
- Modify: `src/lafla_ai_core/runtime/policy.py`

- [ ] **Step 1: Write the failing test**

```python
def test_public_runtime_cleans_bytelevel_and_mojibake_surface(self):
    config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
    output = render_runtime_output("\u0120T\u00c3\u00bcrk\u00c3\u00a7e \u0120k\u00c4\u00b1sa \u0120cevap", config)
    self.assertEqual(output.public_text, "Türkçe kısa cevap")
    self.assertNotIn("possible_mojibake", output.warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='src'; python -m unittest tests.unit.test_runtime_policy.RuntimePolicyTest.test_public_runtime_cleans_bytelevel_and_mojibake_surface -v`

Expected: FAIL because `render_runtime_output` returns byte-level/mojibake surface text.

- [ ] **Step 3: Implement minimal fix**

Route public text and developer raw thinking through `clean_decoded_text`; keep private thinking stripping before public cleanup.

- [ ] **Step 4: Verify**

Run the single test, then `python -m unittest tests.unit.test_runtime_policy -v`.

### Task 2: Central Chat Template Contract

**Files:**
- Create: `src/lafla_ai_core/tokenizer/chat_template.py`
- Create: `tests/unit/test_chat_template.py`
- Modify: `src/lafla_ai_core/data/packing.py`
- Modify: `src/lafla_ai_core/post_training/thinking_sft.py`
- Modify: `tests/unit/test_packing.py`

- [ ] **Step 1: Write failing tests**

Tests cover deterministic role order, control-token rejection, generation prompt rendering, and prompt/response JSONL using the same renderer.

- [ ] **Step 2: Verify red**

Run: `$env:PYTHONPATH='src'; python -m unittest tests.unit.test_chat_template tests.unit.test_packing -v`

Expected: import failure for the new module and/or missing BOS chat formatting.

- [ ] **Step 3: Implement minimal module**

Define `ChatTurn`, role token constants, `render_chat_transcript`, `render_generation_prompt`, and `validate_no_control_tokens`.

- [ ] **Step 4: Wire callers**

Use the shared renderer in `packing.py` and re-export compatible names in `thinking_sft.py`.

### Task 3: Checkpoint Contract

**Files:**
- Create: `src/lafla_ai_core/model/checkpoint_contract.py`
- Create: `tests/unit/test_checkpoint_contract.py`
- Modify: `src/lafla_ai_core/model/checkpoint_io.py`

- [ ] **Step 1: Write failing tests**

Tests reject missing `READY.json`, malformed `READY.json`, `ready=false`, wrong format, and missing required checkpoint files.

- [ ] **Step 2: Verify red**

Run: `$env:PYTHONPATH='src'; python -m unittest tests.unit.test_checkpoint_contract -v`

Expected: import failure until the contract module exists.

- [ ] **Step 3: Implement contract**

Validate `READY.json` content and required files without importing Torch.

- [ ] **Step 4: Wire load path**

Call the contract before `torch.load` in `load_training_checkpoint`.

### Task 4: Artifact Manifest CLI And Colab Plan

**Files:**
- Create: `src/lafla_ai_core/cli/artifact_manifest.py`
- Create: `tests/unit/test_artifact_manifest_cli.py`
- Modify: `src/lafla_ai_core/colab/run_plan.py`
- Modify: `tests/unit/test_colab_run_plan.py`

- [ ] **Step 1: Write failing tests**

Tests verify the CLI writes JSON and the Colab plan writes an artifact manifest before tar archive creation.

- [ ] **Step 2: Implement CLI**

Wrap `observability.artifact_manifest.write_manifest` in a small argparse entrypoint.

### Task 5: Hugging Face Export Package

**Files:**
- Create: `src/lafla_ai_core/export/__init__.py`
- Create: `src/lafla_ai_core/export/hf_package.py`
- Create: `tests/unit/test_hf_package.py`

- [ ] **Step 1: Write failing tests**

Tests use a tiny tokenizer JSON vocab and assert output files: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, and `README.md`.

- [ ] **Step 2: Implement minimal package writer**

Resolve special token ids from tokenizer JSON, write a chat template, and make the package self-describing. Do not claim standard `AutoModelForCausalLM` compatibility without a model-code or weight-name conversion layer.

### Task 6: Groundedness Gate

**Files:**
- Create: `src/lafla_ai_core/evaluation/grounding.py`
- Create: `tests/unit/test_grounding_gate.py`

- [ ] **Step 1: Write failing tests**

Tests reject factual answers without required evidence markers and pass answers that cite known evidence ids or explicitly state uncertainty.

- [ ] **Step 2: Implement deterministic gate**

This is an eval/release gate, not a magic hallucination cure. It prevents unsupported factual release samples from being marked green.

### Task 7: Verification

Run:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python -m lafla_ai_core.cli.quality_scan --root .
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: quality scan OK, full unittest OK with only Torch-dependent tests skipped if Torch is not installed.
