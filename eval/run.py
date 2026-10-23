"""Compares three ways to train on the demo split, with the same total steps:

  local    each member trains alone for rounds * steps, evaluated on the shared test set
  fed      FFA-LoRA: each round every member trains `steps` from the merged B, then B is averaged
  central  one model trained on the union of all members' data for rounds * steps

Runs the trainer as a subprocess (the same entry point the daemon uses) and
merges with the same rule as the daemon: mean of B weighted by example count,
summed in member order in float64, cast to float32.

  .venv/bin/python eval/run.py --rounds 3 --seeds 1,2,3 [--out eval/results/run.json]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from safetensors.numpy import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "..", "trainer", "trainer.py")
MEMBERS = ["alice", "bob", "carol"]


def run(args):
    out = subprocess.run([sys.executable, TRAINER, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def merge(paths, weights, out):
    total = float(sum(weights))
    tensors = [load_file(p) for p in paths]
    merged = {}
    for key in sorted(tensors[0]):
        acc = np.zeros(tensors[0][key].shape, dtype=np.float64)
        for t, w in zip(tensors, weights):
            acc += t[key].astype(np.float64) * (w / total)
        merged[key] = acc.astype(np.float32)
    save_file(merged, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(HERE, "config.json"))
    p.add_argument("--data", default="data/demo")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--out")
    args = p.parse_args()

    base_cfg = json.load(open(args.config, encoding="utf-8"))
    test = os.path.join(args.data, "test.jsonl")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in [int(s) for s in args.seeds.split(",")]:
            cfg = {**base_cfg, "seed": seed}
            cfg_path = os.path.join(tmp, f"cfg-{seed}.json")
            json.dump(cfg, open(cfg_path, "w"))
            long_cfg = {**cfg, "steps": cfg["steps"] * args.rounds}
            long_path = os.path.join(tmp, f"cfg-long-{seed}.json")
            json.dump(long_cfg, open(long_path, "w"))
            row = {"seed": seed}

            local = {}
            for m in MEMBERS:
                b = os.path.join(tmp, f"local-{m}-{seed}.safetensors")
                run(["train", "--config", long_path, "--data", os.path.join(args.data, f"{m}.jsonl"), "--out", b, "--round", "1"])
                local[m] = run(["eval", "--config", cfg_path, "--data", test, "--adapter", b])["accuracy"]
            row["local"] = local
            row["localMean"] = round(sum(local.values()) / len(local), 4)

            merged = None
            fed_curve = []
            for r in range(1, args.rounds + 1):
                outs, weights = [], []
                for m in MEMBERS:
                    b = os.path.join(tmp, f"fed-{m}-{seed}-{r}.safetensors")
                    train_args = ["train", "--config", cfg_path, "--data", os.path.join(args.data, f"{m}.jsonl"), "--out", b, "--round", str(r)]
                    if merged:
                        train_args += ["--init", merged]
                    res = run(train_args)
                    outs.append(b)
                    weights.append(res["examples"])
                merged = os.path.join(tmp, f"fed-merged-{seed}-{r}.safetensors")
                merge(outs, weights, merged)
                fed_curve.append(run(["eval", "--config", cfg_path, "--data", test, "--adapter", merged])["accuracy"])
            row["fed"] = fed_curve[-1]
            row["fedByRound"] = fed_curve

            b = os.path.join(tmp, f"central-{seed}.safetensors")
            run(["train", "--config", long_path, "--data", os.path.join(args.data, "all.jsonl"), "--out", b, "--round", "1"])
            row["central"] = run(["eval", "--config", cfg_path, "--data", test, "--adapter", b])["accuracy"]
            results.append(row)
            print(json.dumps(row), flush=True)

    summary = {
        "config": base_cfg,
        "rounds": args.rounds,
        "runs": results,
        "mean": {k: round(float(np.mean([r[k] for r in results])), 4) for k in ["localMean", "fed", "central"]},
    }
    print(json.dumps(summary["mean"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
