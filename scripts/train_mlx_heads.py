#!/usr/bin/env python3
"""MLX heads-only 训练：backbone 冻结，只训决策头。

流程：
  1. 用 train_pipeline_decisions 的 loader 读 JSONL（gold_distribution 目标，零格式漂移）
  2. backbone 前向一次，缓存 train+dev 全部 path 的 leaves
  3. nn.value_and_grad + AdamW 只优化 MLXHeads，loss = 整题分布 CE（与 PyTorch 端一致）
  4. dev-NLL 选最优（同官方 dev target cross-entropy 选择），存 best heads + 日志

用法（maze local atomic，复刻官方配置意图）：
  .venv-mlx/bin/python scripts/train_mlx_heads.py \
    --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
    --output-dir runs_mlx/local_atomic --steps 300 --batch-questions 16 \
    --init-heads checkpoint --eval-every 50 --max-length 2048 --seed 17
"""
import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer

from mlx_decision_model import (
    MLXDecisionModel,
    MLXHeads,
    _unflatten,
    init_heads_random,
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
    p.add_argument("--input", required=True, help="JSONL 文件或含 split.jsonl 的目录")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-questions", type=int, default=16)
    p.add_argument("--head-lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-train", type=int, default=0, help="0=全部")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--backbone-chunk-paths", type=int, default=4)
    p.add_argument("--init-heads", choices=["random", "checkpoint"], default="checkpoint")
    p.add_argument("--eval-every", type=int, default=50)
    a = p.parse_args()

    random.seed(a.seed)
    mx.random.seed(a.seed)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(a.checkpoint_dir).expanduser().resolve(strict=True)
    run_config = json.loads((ckpt / "config.json").read_text())
    backbone_cfg = json.loads((ckpt / "backbone_config" / "config.json").read_text())
    hidden = backbone_cfg["hidden_size"]
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"),
                                             local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    pipe = load_pipeline_module()
    t0 = time.perf_counter()
    examples, _audit = pipe.load_training_examples(a.input, tokenizer, a.max_length)

    def usable(split):
        return [ex for ex in examples
                if ex["split"] == split
                and pipe.target_for(ex, "gold_distribution") is not None
                and max(map(len, ex["leaf_tokens"])) <= a.max_length]

    trainable = usable("train")
    if a.max_train > 0:
        trainable = trainable[: a.max_train]
    devable = usable("dev")
    if not trainable:
        raise ValueError("没有可用的 gold_distribution 训练样本")
    print(json.dumps({
        "train_examples": len(trainable), "dev_examples": len(devable),
        "train_by_type": {t: sum(1 for e in trainable if e["type"] == t)
                          for t in ("boolean", "choice", "score")},
        "load_seconds": round(time.perf_counter() - t0, 2)}), flush=True)

    # backbone 搬运冻结；heads 按 init-heads 初始化
    model = MLXDecisionModel(backbone_cfg, run_config["set_head"])
    backbone_flat, _, _ = split_checkpoint_weights(str(ckpt / "best.safetensors"))
    model.backbone.update(_unflatten(backbone_flat))
    if a.init_heads == "checkpoint":
        load_heads_from_checkpoint(model.heads, str(ckpt / "best.safetensors"))
    else:
        init_heads_random(model.heads, hidden, seed=a.seed)
    model.eval()
    mx.eval(model.parameters())

    # 缓存 leaves（train + dev）
    import numpy as _np
    t1 = time.perf_counter()

    def cache_leaves(subset):
        paths = [ids for ex in subset for ids in ex["leaf_tokens"]]
        leaves = _np.zeros((len(paths), hidden), dtype=_np.float32)
        for s in range(0, len(paths), a.backbone_chunk_paths):
            chunk = paths[s : s + a.backbone_chunk_paths]
            width = max(map(len, chunk))
            tok = [[ids[i] if i < len(ids) else pad_id for i in range(width)] for ids in chunk]
            tokens = mx.array(tok, dtype=mx.uint32)
            valid = mx.array([[i < len(ids) for i in range(width)] for ids in chunk])
            out = model.forward_paths(tokens, valid)
            mx.eval(out)
            leaves[s : s + len(chunk)] = _np.array(out)
        offsets, off = [], 0
        for ex in subset:
            offsets.append(off)
            off += len(ex["leaf_tokens"])
        return leaves, offsets

    train_leaves, train_offsets = cache_leaves(trainable)
    print(json.dumps({"cache": f"train {len(trainable)} ex done",
                      "seconds": round(time.perf_counter() - t1, 2)}), flush=True)
    dev_leaves, dev_offsets = cache_leaves(devable) if devable else (None, None)
    print(json.dumps({"cache_seconds": round(time.perf_counter() - t1, 2),
                      "train_paths": int(sum(len(e["leaf_tokens"]) for e in trainable)),
                      "dev_paths": int(sum(len(e["leaf_tokens"]) for e in devable))}), flush=True)

    heads = model.heads
    optimizer = optim.AdamW(learning_rate=a.head_lr)

    def pack_batch(sel, subset, leaves, offsets):
        batch = [subset[i] for i in sel]
        kmax = max(len(ex["candidate_ids"]) for ex in batch)
        hn = _np.zeros((len(batch), kmax, hidden), dtype=_np.float32)
        vn = _np.zeros((len(batch), kmax), dtype=bool)
        tn = _np.zeros((len(batch), kmax), dtype=_np.float32)
        types = []
        for bi, (gi, ex) in enumerate(zip(sel, batch)):
            n = len(ex["leaf_tokens"])
            hn[bi, :n] = leaves[offsets[gi] : offsets[gi] + n]
            vn[bi, : len(ex["candidate_ids"])] = True
            tn[bi, : len(ex["candidate_ids"])] = pipe.target_for(ex, "gold_distribution")
            types.append(ex["type"])
        return mx.array(hn), mx.array(vn), types, mx.array(tn)

    def loss_fn(heads_mod, h, valid, types, targets):
        logits = heads_mod(h, valid, types)
        logp = nn.log_softmax(mx.where(valid, logits, float("-inf")), axis=-1)
        return -(mx.where(valid, targets * logp, 0.0)).sum(axis=-1).mean()

    def eval_step(h, valid, targets, types):
        logits = heads(h, valid, types)
        logp = nn.log_softmax(mx.where(valid, logits, float("-inf")), axis=-1)
        return -(mx.where(valid, targets * logp, 0.0)).sum(axis=-1).mean()

    def dev_nll():
        if not devable:
            return None
        losses = []
        for b in range(0, len(devable), a.batch_questions):
            sel = list(range(b, min(b + a.batch_questions, len(devable))))
            h, valid, types, targets = pack_batch(sel, devable, dev_leaves, dev_offsets)
            mx.eval(h, valid, targets)
            losses.append(float(_np.array(eval_step(h, valid, targets, types))))
        return sum(losses) / len(losses)

    loss_and_grad = nn.value_and_grad(heads, loss_fn)
    idxs = list(range(len(trainable)))
    log = []
    best = {"nll": dev_nll(), "step": 0}
    from mlx.utils import tree_flatten, tree_unflatten
    best_params = dict(tree_flatten(heads.parameters()))
    print(json.dumps({"init_dev_nll": best["nll"]}), flush=True)
    t2 = time.perf_counter()
    for step in range(a.steps):
        random.shuffle(idxs)
        step_losses = []
        for b in range(0, len(idxs), a.batch_questions):
            sel = idxs[b : b + a.batch_questions]
            h, valid, types, targets = pack_batch(sel, trainable, train_leaves, train_offsets)
            loss, grads = loss_and_grad(heads, h, valid, types, targets)
            optimizer.update(heads, grads)
            mx.eval(heads.parameters(), optimizer.state)
            mx.eval(loss)
            step_losses.append(float(_np.array(loss)))
        avg = sum(step_losses) / len(step_losses)
        entry = {"step": step + 1, "loss": avg}
        if a.eval_every > 0 and (step + 1) % a.eval_every == 0:
            nll = dev_nll()
            entry["dev_nll"] = nll
            if nll is not None and nll < best["nll"]:
                best = {"nll": nll, "step": step + 1}
                best_params = dict(tree_flatten(heads.parameters()))
        log.append(entry)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in entry.items()}), flush=True)
    print(json.dumps({"train_seconds": round(time.perf_counter() - t2, 2), "best": best}), flush=True)

    heads.update(tree_unflatten(list(best_params.items())))
    mx.eval(heads.parameters())
    mx.save_safetensors(str(out_dir / "heads_best.safetensors"), dict(tree_flatten(heads.parameters())))
    (out_dir / "train_log.json").write_text(
        json.dumps({"args": vars(a), "log": log, "best": best}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved": str(out_dir / "heads_best.safetensors"),
                      "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"],
                      "best": best}), flush=True)


if __name__ == "__main__":
    main()
