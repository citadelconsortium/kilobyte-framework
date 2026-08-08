#!/usr/bin/env python3
"""Kilobyte training notebook — runs on a Kaggle GPU session, fully offline.

Kaggle notebooks have no internet unless the account is phone-verified, so this pipeline
does not rely on it: the base model is attached as a Kaggle model input, and it uses only
packages present on Kaggle's GPU image (torch, transformers, peft, datasets). A 1.5B model
fine-tunes comfortably in bf16 with a LoRA adapter, so no 4-bit/bitsandbytes or Unsloth is
needed.

GGUF conversion is deliberately not done here — it needs llama.cpp, which needs internet
to fetch. This notebook outputs the merged Hugging Face weights; conversion to
kilobyte.gguf happens afterward on a machine that has llama.cpp (see convert_gguf.sh).

Inputs on Kaggle:
    /kaggle/input/qwen2.5/transformers/1.5b-instruct/1   base model (attached)
    /kaggle/input/kilobyte-sft/kilobyte-sft.jsonl        dataset (attached)
    /kaggle/input/kilobyte-sft/config.json               config (bundled)
Outputs:
    /kaggle/working/output/merged/                       merged HF weights
    /kaggle/working/output/lora-adapter/                 the LoRA adapter
    /kaggle/working/output/train_metrics.json            metrics
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


def log(msg: str) -> None:
    print(f"[kilobyte] {msg}", flush=True)


def load_config() -> dict:
    candidates = [Path("config.json"), Path(__file__).parent / "config.json"]
    candidates += [Path(p) for p in glob.glob("/kaggle/input/*/config.json")]
    candidates.append(Path(__file__).parent / "config.example.json")
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    raise SystemExit("config.json not found")


def on_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


def resolve_base_model(config: dict) -> str:
    """Find the attached base model. On Kaggle it is mounted read-only under /kaggle/input;
    the exact version subdirectory is globbed so a version bump does not break the path."""
    if not on_kaggle():
        return config.get("local_base_model", config["base_model"])
    for pattern in (
        "/kaggle/input/qwen2.5/transformers/1.5b-instruct/*",
        "/kaggle/input/*/transformers/1.5b-instruct/*",
        "/kaggle/input/*/**/config.json",
    ):
        hits = sorted(glob.glob(pattern, recursive=True))
        for hit in hits:
            path = Path(hit)
            root = path.parent if path.name == "config.json" else path
            if (root / "config.json").is_file() and any(root.glob("*.safetensors")):
                log(f"using attached base model at {root}")
                return str(root)
    raise SystemExit("attached base model not found under /kaggle/input; attach qwen-lm/qwen2.5/transformers/1.5b-instruct")


def resolve_paths(config: dict) -> tuple[Path, Path]:
    if on_kaggle():
        slug = config["kaggle"]["dataset_slug"]
        data_dir = Path(f"/kaggle/input/{slug}")
        out_dir = Path("/kaggle/working/output")
    else:
        data_dir = Path("data")
        out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir


def _conversation_to_messages(conv: dict) -> list[dict]:
    """Flatten a spec conversation into chat-template messages. Tool calls are folded into
    the assistant text and tool results into a user turn, so the template renders cleanly
    on tokenizers that only know system/user/assistant."""
    messages = []
    for message in conv["messages"]:
        role = message["role"]
        content = message.get("content", "")
        if role == "assistant" and message.get("tool_calls"):
            calls = "\n".join(
                json.dumps({"tool": c["name"], "arguments": c["arguments"]}, ensure_ascii=False)
                for c in message["tool_calls"]
            )
            content = (content + "\n" + calls).strip()
        if role == "tool":
            role, content = "user", f"[tool result] {content}"
        messages.append({"role": role, "content": content})
    return messages


def render_examples(path: Path, tokenizer, max_len: int) -> "list[dict]":
    """Render each conversation to text via the chat template, then tokenise the whole
    sequence. Robust across tokenizer versions: the template is rendered to a string
    first, then tokenised once, rather than per turn."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        conv = json.loads(line)
        text = tokenizer.apply_chat_template(
            _conversation_to_messages(conv), tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        if ids:
            # Full-sequence supervised fine-tuning: labels mirror inputs. Effective for
            # teaching identity, persona and tool-call format on a focused dataset.
            rows.append({"input_ids": ids, "labels": list(ids)})
    return rows


def _disable_optional_backends() -> None:
    """Kaggle ships an old torchao that peft's LoRA dispatch rejects with an ImportError
    even though we do not use it. Force the availability checks to report absent so the
    plain LoRA path is taken. No internet to upgrade the package, so this is the fix."""
    try:
        import peft.import_utils as iu
        for name in ("is_torchao_available", "is_aqlm_available", "is_eetq_available", "is_hqq_available"):
            if hasattr(iu, name):
                setattr(iu, name, lambda *a, **k: False)
        import peft.tuners.lora.torchao as lt
        if hasattr(lt, "is_torchao_available"):
            lt.is_torchao_available = lambda *a, **k: False
    except Exception as exc:  # noqa: BLE001 - best-effort guard
        log(f"could not patch optional backend checks: {exc}")


def train(config: dict, data_dir: Path, out_dir: Path) -> Path:
    import torch
    _disable_optional_backends()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cuda = torch.cuda.is_available()
    if cuda:
        try:
            # Sanity: Kaggle sometimes assigns a GPU whose arch the preinstalled torch was
            # not built for ("no kernel image available"). Prove a kernel actually runs.
            _ = (torch.zeros(1, device="cuda") + 1).item()
            log(f"CUDA ok: {torch.cuda.get_device_name(0)}")
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: GPU present but unusable ({str(exc)[:120]}); falling back to CPU fp32.")
            cuda = False
    if not cuda:
        log("Training on CPU in fp32 — slower but reliable; fine for a small dataset.")
    base = resolve_base_model(config)
    tc = config["train"]
    log(f"loading tokenizer from {base}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("loading base model (this is the slow load step)...")
    dtype = torch.float16 if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=("auto" if cuda else None),
    )
    if not cuda:
        model = model.to("cpu")
    log("base model loaded")
    model.config.use_cache = False
    if tc.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"], task_type="CAUSAL_LM", bias="none",
    ))
    model.print_trainable_parameters()

    rows = render_examples(data_dir / "kilobyte-sft.jsonl", tokenizer, config["max_seq_len"])
    log(f"prepared {len(rows)} training rows")

    def collate(batch):
        pad = tokenizer.pad_token_id
        width = max(len(b["input_ids"]) for b in batch)
        ids, labs, mask = [], [], []
        for b in batch:
            seq, lab = list(b["input_ids"]), list(b["labels"])
            n = len(seq)
            ids.append(seq + [pad] * (width - n))
            labs.append(lab + [-100] * (width - n))
            mask.append([1] * n + [0] * (width - n))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(mask),
        }

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        num_train_epochs=tc["epochs"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        fp16=cuda,
        use_cpu=not cuda,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=tc["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=rows, data_collator=collate)
    result = trainer.train()
    (out_dir / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    log(f"training done: {result.metrics}")

    adapter = out_dir / "lora-adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    log("merging adapter into the base weights")
    log("training complete; merging adapter")
    merged_model = model.merge_and_unload()
    merged = out_dir / "merged"
    log("saving merged weights (~3 GB, this takes a few minutes)")
    merged_model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer.save_pretrained(str(merged))
    log("merged weights saved")
    (out_dir / "MERGED_READY.txt").write_text(
        "Merged HF weights are ready. Convert to GGUF with convert_gguf.sh on a machine with llama.cpp.\n"
    )
    return merged


def main() -> int:
    config = load_config()
    data_dir, out_dir = resolve_paths(config)
    merged = train(config, data_dir, out_dir)
    log(f"merged weights at {merged}")
    log("NOT converting to GGUF here (needs llama.cpp/internet). Run convert_gguf.sh next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
