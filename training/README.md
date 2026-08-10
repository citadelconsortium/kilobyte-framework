# Kilobyte training pipeline

> **Maintainer-only.** End users never train. Installing Kilo downloads one prebuilt,
> checksum-pinned `kilobyte.gguf`; this pipeline is only how the maintainer *produces*
> or updates that single canonical brain. There is one brain, trained once, shared by
> every install.

This directory builds the one canonical Kilobyte brain: a QLoRA fine-tune of a small,
strong open-weight instruct base into `kilobyte.gguf`. Kaggle is the training factory;
Kilo is the runtime. Training never deploys automatically — a new brain is a **candidate**
until it passes evaluation and is explicitly promoted.

```
dataset → Kaggle → train → validate → merge → GGUF → quantise → test
        → Kilo integration test → resource test → regression → PROMOTE → post-promotion test
```

## What runs where

| Stage | Where | Script |
|---|---|---|
| Build & validate the dataset | your machine (CPU) | `build_dataset.py` |
| Push notebook + data, run, fetch output | your machine | `kaggle_run.py` |
| Fine-tune and merge | Kaggle GPU | `kaggle_notebook.py` |
| Convert merged weights to pinned Q4_K_M GGUF | Kaggle CPU | `kaggle_convert_run.py` |
| Evaluate a candidate GGUF | your machine or Kaggle | `evaluate.py` |
| Exercise the candidate through Kilo RPC | isolated Kilo daemon | `framework_evaluate.py` |
| Stage / promote / rollback the brain | Kilo host | `kilo brain …` (see `brains.py`) |

CPU-only preparation is done before any GPU session starts, so Kaggle's free GPU time is
spent only on training — not on validating or formatting data.

## 1. Configure

```bash
cp config.example.json config.json
# edit: base model, hyperparameters, Kaggle username/slug
```

The current replacement base is IBM Granite 4.1 3B, selected as a different and materially
stronger brain than the former Qwen2.5 1.5B model. Its official model card reports native
function calling, strong instruction following and code performance, while a Q4_K_M build
remains practical on Kilo's 4 GB CPU-only target. The training data is rendered through
the model's native tool template; tool calls are never flattened into chat text.

## 2. Authenticate with Kaggle

The pipeline uses the official Kaggle API. Provide credentials the supported way — never
in code, notebooks, git, or the dataset:

```bash
# Newer KGAT_-prefixed token (kaggle.com → Settings → Account → API):
export KAGGLE_API_TOKEN=KGAT_...
# or the classic pair:
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-key
# or place kaggle.json at ~/.kaggle/kaggle.json with mode 600
```

A `KGAT_` token must go in `KAGGLE_API_TOKEN`, not in `kaggle.json` — that is the most
common cause of a 401.

`kaggle_run.py` verifies authentication before submitting anything and never prints the
key.

## 3. Build the dataset (CPU)

```bash
python build_dataset.py --out data/kilobyte-sft.jsonl
```

This validates every example against `dataset_spec.md`, deduplicates, checks the
train/val split, and writes chat-formatted JSONL. A small curated seed set lives in
`seed/`; extend it following the spec's domain distribution.

## 4. Train on Kaggle

```bash
python kaggle_run.py --config config.json
```

Uploads the dataset as a Kaggle dataset, pushes `kaggle_notebook.py` as a notebook with
GPU enabled, starts it, polls status, and downloads the merged weights, LoRA adapter and
metrics. GGUF conversion and Q4_K_M quantisation happen afterward with the pinned
`llama.cpp` converter.
The pipeline requests `NvidiaTeslaT4` explicitly because Kaggle's current default P100 is
not supported by the default PyTorch CUDA build; the requested accelerator is recorded in
`config.json` and printed at submission.

## 5. Evaluate the candidate

```bash
python evaluate.py --model output/kilobyte-candidate.gguf --report output/eval.json
```

Runs the fixed evaluation suite (identity, reasoning, coding, Linux, security reasoning,
tool-calling, conciseness) against the actual GGUF. A candidate that is conversationally
fine but unreliable at tool calls does not pass.

Then start an isolated daemon with `KILOBYTE_MODEL_PATH` set to the candidate and run:

```bash
python framework_evaluate.py --socket /tmp/kilo-candidate/kilobyte.sock \
  --report output/framework-eval.json
```

This is the promotion gate for real `system_info`, file write/read and approval flow,
memory/recall, skill save/list, multi-tool research, result follow-through, and suppression
of raw tool protocol in final client text.

## 6. Promote (only if it passed)

On the Kilo host, with the candidate GGUF in place:

```bash
kilo brain stage output/kilobyte-candidate.gguf
kilo brain promote          # current → previous, candidate → current, atomically
kilo brain status
# if the post-promotion smoke test fails:
kilo brain rollback
```

`brains.py` guarantees the previous brain always survives, so a bad promotion is one
command to undo.

## Artifacts kept per release

`kilobyte-4.1-3b-q4_k_m.gguf` (also published under the stable installer alias
`kilobyte.gguf`), the merged HF weights, the LoRA adapter,
training/validation metrics, evaluation results, the exact base-model revision, the
dataset version, conversion and quantisation commands, dependency versions, and SHA-256
hashes — so every Kilobyte release is auditable and reproducible.
