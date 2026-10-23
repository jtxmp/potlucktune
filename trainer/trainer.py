"""PotluckTune trainer: one member's local step of federated LoRA (FFA-LoRA).

Every member derives the same frozen LoRA A matrices from the team seed and
trains only B. Because A is shared, the average of the members' B matrices
gives exactly the average of their weight updates (B @ A), so a plain
weighted mean is an exact merge.

Called by the Node daemon as a subprocess. Prints one JSON line on stdout.

  trainer.py train --config cfg.json --data local.jsonl --init B.safetensors --out B_new.safetensors
  trainer.py eval  --config cfg.json --data test.jsonl [--adapter B.safetensors]
  trainer.py a-hash --config cfg.json
  trainer.py export --config cfg.json --adapter B.safetensors --out adapter-dir/
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

PROMPT = "Classify the customer message into one intent.\nMessage: {text}\nIntent:"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["model", "rank", "alpha", "targets", "seed", "lr", "steps", "batch", "labels"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"config: missing {', '.join(missing)}")
    return cfg


def read_jsonl(path, labels):
    rows = []
    allowed = set(labels)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or row.get("label") not in allowed:
                raise SystemExit(f"{path}:{n}: each line needs a string 'text' and a 'label' from the team's label list")
            rows.append(row)
    return rows


def frozen_a(seed, name, rank, in_features):
    """Deterministic A for one module: uniform(-1/sqrt(in), 1/sqrt(in)).

    Generator(PCG64).random() has a stable bit stream across NumPy versions,
    unlike the distribution helpers, so every member gets identical bytes.
    """
    digest = hashlib.sha256(f"{seed}/{name}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
    bound = 1.0 / math.sqrt(in_features)
    return ((rng.random((rank, in_features)) * 2.0 - 1.0) * bound).astype(np.float32)


def load_model(cfg, device):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
    lora = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["targets"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)

    a_hash = hashlib.sha256()
    b_params = {}
    for name, module in sorted(model.named_modules()):
        if not hasattr(module, "lora_A") or "default" not in getattr(module, "lora_A", {}):
            continue
        key = module_key(name)
        lin_a = module.lora_A["default"]
        lin_b = module.lora_B["default"]
        a = frozen_a(cfg["seed"], key, cfg["rank"], lin_a.in_features)
        a_hash.update(key.encode())
        a_hash.update(a.tobytes())
        with torch.no_grad():
            lin_a.weight.copy_(torch.from_numpy(a))
            lin_b.weight.zero_()
        lin_a.weight.requires_grad_(False)
        lin_a.weight.data = lin_a.weight.data.float()
        lin_b.weight.data = lin_b.weight.data.float()
        lin_b.weight.requires_grad_(True)
        b_params[key] = lin_b.weight
    for name, p in model.named_parameters():
        if not any(p is b for b in b_params.values()):
            p.requires_grad_(False)
    return tok, model, b_params, a_hash.hexdigest()


def module_key(name):
    # "base_model.model.model.layers.0.self_attn.q_proj" -> "model.layers.0.self_attn.q_proj"
    prefix = "base_model.model."
    return name[len(prefix):] if name.startswith(prefix) else name


def load_b(path, b_params):
    import torch
    from safetensors.numpy import load_file

    tensors = load_file(path)
    expected = {f"{k}.lora_B" for k in b_params}
    if set(tensors) != expected:
        raise SystemExit(f"{path}: adapter does not match this model and config")
    with torch.no_grad():
        for key, param in b_params.items():
            t = tensors[f"{key}.lora_B"]
            if tuple(t.shape) != tuple(param.shape):
                raise SystemExit(f"{path}: wrong shape for {key}")
            param.copy_(torch.from_numpy(t))


def save_b(path, b_params):
    from safetensors.numpy import save_file

    tensors = {f"{k}.lora_B": p.detach().float().cpu().numpy() for k, p in sorted(b_params.items())}
    tmp = f"{path}.{os.getpid()}.tmp"
    save_file(tensors, tmp)
    os.replace(tmp, path)


def encode_pair(tok, text, label, max_len):
    prompt_ids = tok(PROMPT.format(text=text), add_special_tokens=False)["input_ids"]
    target_ids = tok(" " + label, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    prompt_ids = prompt_ids[: max(1, max_len - len(target_ids))]
    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return ids, labels


def collate(tok, pairs, device):
    import torch

    width = max(len(ids) for ids, _ in pairs)
    input_ids, labels, mask = [], [], []
    for ids, lab in pairs:
        pad = width - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(labels, device=device),
        torch.tensor(mask, device=device),
    )


def cmd_train(args, cfg, device):
    import torch

    rows = read_jsonl(args.data, cfg["labels"])
    if not rows:
        raise SystemExit(f"{args.data}: no examples")
    tok, model, b_params, a_hash = load_model(cfg, device)
    if args.init:
        load_b(args.init, b_params)
    torch.manual_seed(cfg["seed"] + args.round)
    rng = np.random.default_rng(cfg["seed"] + args.round)
    max_len = cfg.get("maxLen", 96)
    pairs = [encode_pair(tok, r["text"], r["label"], max_len) for r in rows]
    opt = torch.optim.AdamW(list(b_params.values()), lr=cfg["lr"], weight_decay=0.0)
    model.train()
    if cfg.get("gradientCheckpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    start = time.time()
    losses = []
    order = rng.permutation(len(pairs))
    pos = 0
    for _ in range(cfg["steps"]):
        idx = []
        while len(idx) < cfg["batch"]:
            if pos == len(order):
                order = rng.permutation(len(pairs))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        input_ids, labels, mask = collate(tok, [pairs[i] for i in idx], device)
        out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
    save_b(args.out, b_params)
    tail = losses[-max(1, len(losses) // 5):]
    result = {
        "examples": len(rows),
        "steps": cfg["steps"],
        "loss": round(sum(tail) / len(tail), 4),
        "aHash": a_hash,
        "seconds": round(time.time() - start, 1),
        "device": device,
    }
    if device == "cuda":
        result["peakMemMB"] = round(torch.cuda.max_memory_allocated() / 2**20)
    print(json.dumps(result))


def cmd_eval(args, cfg, device):
    import torch

    rows = read_jsonl(args.data, cfg["labels"])
    tok, model, b_params, _ = load_model(cfg, device)
    if args.adapter:
        load_b(args.adapter, b_params)
    model.eval()
    labels = cfg["labels"]
    max_len = cfg.get("maxLen", 96)
    correct = 0
    per_label = {l: [0, 0] for l in labels}
    batch = cfg.get("evalBatch", 48)
    start = time.time()
    with torch.no_grad():
        for row in rows:
            pairs = [encode_pair(tok, row["text"], l, max_len) for l in labels]
            scores = []
            for i in range(0, len(pairs), batch):
                chunk = pairs[i : i + batch]
                input_ids, lab, mask = collate(tok, chunk, device)
                logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                target = lab[:, 1:]
                keep = target != -100
                gathered = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                # Mean log-prob per label token, so longer label names are not penalised.
                s = (gathered * keep).sum(-1) / keep.sum(-1)
                scores.extend(s.tolist())
            pred = labels[int(np.argmax(scores))]
            per_label[row["label"]][1] += 1
            if pred == row["label"]:
                correct += 1
                per_label[row["label"]][0] += 1
    print(
        json.dumps(
            {
                "examples": len(rows),
                "accuracy": round(correct / len(rows), 4) if rows else 0.0,
                "perLabel": {l: round(c / t, 3) if t else None for l, (c, t) in per_label.items()},
                "seconds": round(time.time() - start, 1),
            }
        )
    )


def cmd_a_hash(args, cfg, device):
    _, _, _, a_hash = load_model(cfg, "cpu")
    print(json.dumps({"aHash": a_hash}))


def cmd_export(args, cfg, device):
    """Writes a standard PEFT adapter directory (frozen A plus merged B)."""
    tok, model, b_params, a_hash = load_model(cfg, "cpu")
    load_b(args.adapter, b_params)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(json.dumps({"out": args.out, "aHash": a_hash, "model": cfg["model"]}))


def main():
    p = argparse.ArgumentParser(prog="trainer.py")
    p.add_argument("command", choices=["train", "eval", "a-hash", "export"])
    p.add_argument("--config", required=True)
    p.add_argument("--data")
    p.add_argument("--init")
    p.add_argument("--out")
    p.add_argument("--adapter")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    cfg = load_config(args.config)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "train":
        if not args.data or not args.out:
            raise SystemExit("train needs --data and --out")
        cmd_train(args, cfg, device)
    elif args.command == "eval":
        if not args.data:
            raise SystemExit("eval needs --data")
        cmd_eval(args, cfg, device)
    elif args.command == "export":
        if not args.adapter or not args.out:
            raise SystemExit("export needs --adapter and --out")
        cmd_export(args, cfg, device)
    else:
        cmd_a_hash(args, cfg, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
