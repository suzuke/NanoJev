#!/usr/bin/env python3
"""MLX heads 评测：test/ood 的 accuracy / scalar-Brier / NLL（对齐 docs/ATOMIC_PLANNING.md 口径）。

用法：
  .venv-mlx/bin/python scripts/evaluate_mlx_heads.py \
    --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
    --heads runs_mlx/local_atomic/heads_best.safetensors --max-length 2048
  --heads 也可以是 released（用 checkpoint 自带 heads，即训练起点）。
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as _np
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

from mlx_decision_model import (
    MLXDecisionModel,
    _unflatten,
    load_heads_from_checkpoint,
    split_checkpoint_weights,
)


def load_pipeline_module():
    path = Path(__file__).with_name("train_pipeline_decisions.py")
    spec = importlib.util.spec_from_file_location("nanojev_pipeline_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--heads", default="released", help="heads.safetensors 或 released")
    p.add_argument("--trainable", default=None,
                   help="train_mlx_lora.py 存的 trainable_best.safetensors（heads+LoRA，需先 apply_lora）")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--splits", default="test,ood")
    p.add_argument("--batch-questions", type=int, default=16)
    a = p.parse_args()

    ckpt = Path(a.checkpoint_dir).expanduser().resolve(strict=True)
    run_config = json.loads((ckpt / "config.json").read_text())
    backbone_cfg = json.loads((ckpt / "backbone_config" / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"),
                                             local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = load_pipeline_module()
    examples, _ = pipe.load_training_examples(a.input, tokenizer, a.max_length)
    model = MLXDecisionModel(backbone_cfg, run_config["set_head"])
    backbone_flat, _, _ = split_checkpoint_weights(str(ckpt / "best.safetensors"))
    model.backbone.update(_unflatten(backbone_flat))
    if a.heads == "released":
        load_heads_from_checkpoint(model.heads, str(ckpt / "best.safetensors"))
    else:
        saved = dict(mx.load(a.heads))
        model.heads.update(tree_unflatten(list(saved.items())))
    if a.trainable:
        from mlx_decision_model import apply_lora, freeze_for_lora_heads
        apply_lora(model, rank=a.lora_rank, alpha=a.lora_alpha)
        saved = dict(mx.load(a.trainable))
        model.update(tree_unflatten(list(saved.items())))
    model.eval()
    mx.eval(model.parameters())

    out = {}
    for split in a.splits.split(","):
        subset = [ex for ex in examples
                  if ex["split"] == split and ex["gold_index"] is not None
                  and max(map(len, ex["leaf_tokens"])) <= a.max_length]
        n_correct, n_total, brier_sum, nll_sum = 0, 0, 0.0, 0.0
        for b in range(0, len(subset), a.batch_questions):
            batch = subset[b : b + a.batch_questions]
            logits, valid = model(batch, tokenizer.pad_token_id)
            mx.eval(logits)
            ln = _np.array(logits, dtype=_np.float64)
            for ex, row in zip(batch, ln):
                k = len(ex["candidate_ids"])
                scores = row[:k]
                m = scores.max()
                probs = _np.exp(scores - m)
                probs /= probs.sum()
                pred = int(_np.argmax(probs))
                gold = ex["gold_index"]
                n_correct += pred == gold
                n_total += 1
                if ex["type"] == "boolean":
                    p_true, y = float(probs[1]), float(gold)
                    brier_sum += (p_true - y) ** 2
                    nll_sum += -math.log(max(probs[gold], 1e-30))
                else:
                    nll_sum += -math.log(max(probs[gold], 1e-30))
        out[split] = {"n": n_total, "accuracy": n_correct / n_total,
                      "scalar_brier": brier_sum / n_total, "nll": nll_sum / n_total}
        print(json.dumps({"split": split, **out[split]}), flush=True)


if __name__ == "__main__":
    main()
