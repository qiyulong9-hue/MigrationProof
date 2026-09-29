---
name: migrationproof-ascend-audit
description: Use when a university AI laboratory needs to validate Qwen 3.5 migration, fine-tuning evidence, accuracy gates, or comparable performance optimization on Huawei Ascend NPU.
---

# MigrationProof Ascend Audit

Audit one declared Qwen 3.5 Ascend migration task. The model may interpret and explain; deterministic Python tools must compute every version, hash, metric, comparison, and final status.

## Required inputs

- A task JSON validated against `configs/task_semifinal_v1.json`.
- An already working official Ascend base environment.
- Immutable model, tokenizer, dataset split, baseline config, and optimized config identities.
- Raw accuracy and performance artifacts with SHA-256 hashes.

If an input is absent or contradictory, preserve it as missing evidence. Never invent a value or silently lower a threshold.

## Workflow

1. Load the project session with `source env/load_project_env.sh`.
2. Validate the task before creating a run.
3. Run `probe`, then `smoke`.
4. Evaluate the fixed application split and apply the declared accuracy gate.
5. Compare an unoptimized Ascend baseline with an optimized Ascend candidate on identical hardware and protocol.
6. Run deterministic acceptance only after accuracy and comparability checks pass.
7. Generate a report and evidence ledger; bind every claim to an artifact path and SHA-256.

Use at least 20 post-warmup measurements for a competition performance claim. A GPU result cannot replace the unoptimized Ascend baseline.

## Status contract

- `pass`: accuracy and comparable performance gates pass with complete evidence.
- `conditional_pass`: the migration core passes but a declared non-core application item remains pending.
- `fail`: a correctness, smoke, accuracy, or performance gate is proven to fail.
- `insufficient_evidence`: a required identity, artifact, repetition count, or fair comparison is missing.

The model may improve the rationale but must not change the deterministic status or issue codes.

## Safety boundary

Do not install or modify drivers, firmware, kernels, CANN, or production services. Preserve historical logs and reports. Write every experiment to a new versioned run directory. Require separate user confirmation before any long training job.

See `README.md` for commands, artifact paths, verified B4 results, and current limitations.
