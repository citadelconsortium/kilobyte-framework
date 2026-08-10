#!/usr/bin/env python3
"""Kilobyte training notebook — runs on a Kaggle GPU session.

The model may be attached as a Kaggle model input or downloaded from its official model
repository when notebook internet is enabled.  A LoRA adapter keeps the 3B run within a
single Kaggle GPU while preserving the base model's general reasoning.

GGUF conversion is deliberately not done here — it needs llama.cpp, which needs internet
to fetch. This notebook outputs the merged Hugging Face weights; conversion to
kilobyte.gguf happens afterward on a machine that has llama.cpp (see convert_gguf.sh).

Inputs on Kaggle:
    ibm-granite/granite-4.1-3b                           official base (downloaded or attached)
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
    for pattern in ("/kaggle/input/*/**/config.json",):
        hits = sorted(glob.glob(pattern, recursive=True))
        for hit in hits:
            path = Path(hit)
            root = path.parent if path.name == "config.json" else path
            if (root / "config.json").is_file() and any(root.glob("*.safetensors")):
                log(f"using attached base model at {root}")
                return str(root)
    log(f"no attached model found; downloading official base {config['base_model']}")
    return config["base_model"]


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


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write text to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "List a directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search files for text.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a command on the local machine.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "system_info", "description": "Inspect live system resources and platform details.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_fetch", "description": "Fetch a web page.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "reference", "description": "Look up Kilo's offline reference bank.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "remember", "description": "Store a durable fact.", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "importance": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "recall", "description": "Recall stored facts.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_history", "description": "Search past Kilo conversations.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "save_skill", "description": "Save a reusable procedure as a Kilo skill.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "when_to_use": {"type": "string"}, "steps": {"type": "string"}}, "required": ["name", "when_to_use", "steps"]}}},
    {"type": "function", "function": {"name": "list_skills", "description": "List the reusable Kilo skills available to this assistant.", "parameters": {"type": "object", "properties": {}}}},
]


def _conversation_to_messages(conv: dict) -> list[dict]:
    """Convert the dataset spec to native OpenAI function-call messages.

    The old pipeline flattened calls into assistant text.  That directly encouraged a
    model to print tool JSON in chat.  Native calls let each model's own chat template
    emit the exact tokens that llama.cpp parses back into framework tool calls.
    """
    messages = []
    pending_ids: list[tuple[str, str]] = []
    call_number = 0
    for message in conv["messages"]:
        role = message["role"]
        content = message.get("content", "")
        if role == "assistant" and message.get("tool_calls"):
            converted = []
            for call in message["tool_calls"]:
                call_id = f"call_{call_number}"
                call_number += 1
                pending_ids.append((call_id, call["name"]))
                converted.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                })
            messages.append({"role": role, "content": content, "tool_calls": converted})
            continue
        if role == "tool":
            call_id, called_name = pending_ids.pop(0) if pending_ids else (f"call_{call_number}", message.get("name", "tool"))
            messages.append({"role": "tool", "name": message.get("name", called_name), "tool_call_id": call_id, "content": content})
        else:
            messages.append({"role": role, "content": content})
    return messages


def render_examples(path: Path, tokenizer, max_len: int) -> "list[dict]":
    """Render native tool conversations and train only on assistant output.

    Granite's template has no Jinja ``generation`` block, so Transformers cannot return
    an assistant mask. Falling back to labels for the whole conversation teaches the model
    to predict user prompts and tool results and caused measurable capability regression.
    Derive exact assistant spans from the same native template instead: the generation
    prefix marks where each assistant answer starts, and the through-message rendering
    marks where it ends.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        conv = json.loads(line)
        messages = _conversation_to_messages(conv)
        template_args = {
            "tools": TOOL_SCHEMAS, "tokenize": True, "add_generation_prompt": False,
            "truncation": True, "max_length": max_len, "return_dict": True,
        }
        rendered = tokenizer.apply_chat_template(messages, **template_args)
        ids = rendered["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            ids = list(ids)
            labels = [-100] * len(ids)
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                before_rendered = tokenizer.apply_chat_template(
                    messages[:index], tools=TOOL_SCHEMAS, tokenize=True,
                    add_generation_prompt=True, return_dict=True,
                )
                through_rendered = tokenizer.apply_chat_template(
                    messages[:index + 1], tools=TOOL_SCHEMAS, tokenize=True,
                    add_generation_prompt=False, return_dict=True,
                )
                before = before_rendered["input_ids"]
                through = through_rendered["input_ids"]
                if before and isinstance(before[0], list):
                    before = before[0]
                if through and isinstance(through[0], list):
                    through = through[0]
                start, end = len(before), min(len(through), len(ids))
                if list(through[:end]) != ids[:end] or start >= end:
                    raise ValueError(f"chat template assistant span mismatch in {conv.get('id', 'unknown')}")
                labels[start:end] = ids[start:end]
            if any(label != -100 for label in labels):
                rows.append({"input_ids": ids, "labels": labels})
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
