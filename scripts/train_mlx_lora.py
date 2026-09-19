#!/usr/bin/env python3
"""MLX LoRA 联训：backbone 挂 LoRA（q/v），heads 全训，其余冻结。

每步都跑 backbone 前向+反向（无 leaves 缓存），适合 M3 统一内存。
用法（maze）：
  .venv-mlx/bin/python scripts/train_mlx_lora.py \
    --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
    --output-dir runs_mlx/local_lora --steps 300 --batch-questions 16 \
    --lora-rank 8 --lora-alpha 16 --lr 1e-4 --eval-every 50 --max-length 2048 --seed 17
"""
import argparse
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
    _unflatten,
    apply_lora,
    freeze_for_lora_heads,
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
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-questions", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--backbone-chunk-paths", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=25,
                   help="每 N 步存一次 trainable_latest.safetensors + train_log.json（防超時白跑）")
    p.add_argument("--init-trainable", default=None,
                   help="从上次存的 trainable_best.safetensors 继续（heads+LoRA）")
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
    examples, _ = pipe.load_training_examples(a.input, tokenizer, a.max_length)

    def usable(split):
        return [ex for ex in examples
                if ex["split"] == split
                and pipe.target_for(ex, "gold_distribution") is not None
                and max(map(len, ex["leaf_tokens"])) <= a.max_length]

    trainable = usable("train")
    if a.max_train > 0:
        trainable = trainable[: a.max_train]
    devable = usable("dev")
    print(json.dumps({
        "train_examples": len(trainable), "dev_examples": len(devable),
        "load_seconds": round(time.perf_counter() - t0, 2)}), flush=True)

    model = MLXDecisionModel(backbone_cfg, run_config["set_head"])
    backbone_flat, _, _ = split_checkpoint_weights(str(ckpt / "best.safetensors"))
    model.backbone.update(_unflatten(backbone_flat))
    load_heads_from_checkpoint(model.heads, str(ckpt / "best.safetensors"))
    apply_lora(model, rank=a.lora_rank, alpha=a.lora_alpha, seed=a.seed)
    n_train = freeze_for_lora_heads(model)
    if a.init_trainable:
        from mlx.utils import tree_unflatten as _tu
        saved = dict(mx.load(a.init_trainable))
        model.update(_tu(list(saved.items())))
        mx.eval(model.parameters())
        print(json.dumps({"resumed_from": a.init_trainable}), flush=True)
    model.eval()
    mx.eval(model.parameters())
    print(json.dumps({"trainable_params": n_train}), flush=True)

    import numpy as _np

    def backbone_leaves(mod, all_ids):
        """分块跑 backbone，返回 (P,H) mx（保持可微分，供 LoRA 联训）。"""
        outs = []
        for s in range(0, len(all_ids), a.backbone_chunk_paths):
            chunk = all_ids[s : s + a.backbone_chunk_paths]
            width = max(map(len, chunk))
            tok = [[ids[i] if i < len(ids) else pad_id for i in range(width)] for ids in chunk]
            tokens = mx.array(tok, dtype=mx.uint32)
            valid = mx.array([[i < len(ids) for i in range(width)] for ids in chunk])
            outs.append(mod.forward_paths(tokens, valid))
        return mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]

    def pack(idxs, batch_ex, leaves, offsets):
        from mlx_decision_model import pack_segments_mx
        k_list, types, t_list = [], [], []
        segs = []
        for gi, ex in zip(idxs, batch_ex):
            n = len(ex["leaf_tokens"])
            k = len(ex["candidate_ids"])
            k_list.append(k)
            s = leaves[offsets[gi] : offsets[gi] + n]
            segs.append(s if isinstance(s, mx.array) else mx.array(s))
            t_list.append(pipe.target_for(ex, "gold_distribution"))
            types.append(ex["type"])
        h, valid = pack_segments_mx(segs, k_list)
        kmax = max(k_list)
        tn = _np.zeros((len(batch_ex), kmax), dtype=_np.float32)
        for bi, (t, k) in enumerate(zip(t_list, k_list)):
            tn[bi, :k] = t
        return h, valid, types, mx.array(tn)

    def batch_loss(mod, batch_ex, idxs):
        all_ids, owners = [], []
        for gi, ex in zip(idxs, batch_ex):
            for ids in ex["leaf_tokens"]:
                all_ids.append(ids)
                owners.append(gi)
        leaves = backbone_leaves(mod, all_ids)
        offsets = {}
        off = 0
        for gi, ex in zip(idxs, batch_ex):
            offsets[gi] = off
            off += len(ex["leaf_tokens"])
        h, valid, types, targets = pack(idxs, batch_ex, leaves, offsets)
        logits = mod.heads(h, valid, types)
        logp = nn.log_softmax(mx.where(valid, logits, float("-inf")), axis=-1)
        return -(mx.where(valid, targets * logp, 0.0)).sum(axis=-1).mean()

    def loss_fn(model_mod, batch_ex, idxs):
        return batch_loss(model_mod, batch_ex, idxs)

    optimizer = optim.AdamW(learning_rate=a.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def dev_nll():
        if not devable:
            return None
        losses = []
        for b in range(0, len(devable), a.batch_questions):
            sel = list(range(b, min(b + a.batch_questions, len(devable))))
            batch = [devable[i] for i in sel]
            mx.eval(loss_val := loss_fn(model, batch, sel))
            losses.append(float(_np.array(loss_val)))
        return sum(losses) / len(losses)

    idxs = list(range(len(trainable)))
    log = []
    best = {"nll": dev_nll(), "step": 0}
    from mlx.utils import tree_flatten, tree_unflatten
    best_params = dict(tree_flatten(model.trainable_parameters()))
    print(json.dumps({"init_dev_nll": best["nll"]}), flush=True)
    t2 = time.perf_counter()
    for step in range(a.steps):
        random.shuffle(idxs)
        step_losses = []
        for b in range(0, len(idxs), a.batch_questions):
            sel = idxs[b : b + a.batch_questions]
            batch = [trainable[i] for i in sel]
            loss, grads = loss_and_grad(model, batch, sel)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            mx.eval(loss)
            step_losses.append(float(_np.array(loss)))
        avg = sum(step_losses) / len(step_losses)
        entry = {"step": step + 1, "loss": avg}
        if a.eval_every > 0 and (step + 1) % a.eval_every == 0:
            nll = dev_nll()
            entry["dev_nll"] = nll
            if nll is not None and nll < best["nll"]:
                best = {"nll": nll, "step": step + 1}
                best_params = dict(tree_flatten(model.trainable_parameters()))
                from mlx.utils import tree_flatten as _tf2
                mx.save_safetensors(str(out_dir / "trainable_best.safetensors"),
                                    dict(_tf2(model.trainable_parameters())))
        log.append(entry)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in entry.items()}), flush=True)
        if a.save_every > 0 and (step + 1) % a.save_every == 0:
            from mlx.utils import tree_flatten as _tf
            mx.save_safetensors(str(out_dir / "trainable_latest.safetensors"),
                                dict(_tf(model.trainable_parameters())))
            (out_dir / "train_log.json").write_text(
                json.dumps({"args": vars(a), "log": log, "best": best,
                            "completed_steps": step + 1},
                           ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train_seconds": round(time.perf_counter() - t2, 2), "best": best}), flush=True)

    # 恢复最优（只含 heads + LoRA）
    flat_now = dict(tree_flatten(model.parameters()))
    for k, v in best_params.items():
        flat_now[k] = v
    model.update(tree_unflatten(list(flat_now.items())))
    mx.eval(model.parameters())
    mx.save_safetensors(str(out_dir / "trainable_best.safetensors"), best_params)
    (out_dir / "train_log.json").write_text(
        json.dumps({"args": vars(a), "log": log, "best": best}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved": str(out_dir / "trainable_best.safetensors"),
                      "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"],
                      "best": best}), flush=True)


if __name__ == "__main__":
    main()
