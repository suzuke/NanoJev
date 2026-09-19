#!/usr/bin/env python3
"""MLX 版 model-guided maze 闭环评测：复用官方 run_exploration + EdgeExplorer，
engine 换成 MLX（backbone 冻结 + LoRA + heads）。

用法：
  .venv-mlx/bin/python scripts/evaluate_mlx_edges_maze.py \
    --episodes results/rollout_pilot_episodes.jsonl \
    --output runs_mlx/edges_maze_local_lora3.json \
    --checkpoint-dir checkpoints/NanoJev \
    --trainable runs_mlx/local_lora3/trainable_best.safetensors \
    --splits test,ood --max-steps 0 --batch-states 2
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

from mlx_decision_model import (
    MLXDecisionModel,
    _unflatten,
    apply_lora,
    split_checkpoint_weights,
)
from evaluate_model_edges_maze import run_exploration


class MLXEdgesEngine:
    """与 DecisionPredictor 同 predict 接口，供 run_exploration 调用。"""

    def __init__(self, checkpoint_dir, trainable, max_length=None,
                 lora_rank=8, lora_alpha=16.0):
        root = Path(checkpoint_dir).expanduser().resolve(strict=True)
        run_config = json.loads((root / "config.json").read_text())
        backbone_cfg = json.loads((root / "backbone_config" / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"),
                                                 local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.limit = run_config.get("max_length", 512) if max_length is None else max_length
        self.model = MLXDecisionModel(backbone_cfg, run_config["set_head"])
        backbone_flat, _, _ = split_checkpoint_weights(str(root / "best.safetensors"))
        self.model.backbone.update(_unflatten(backbone_flat))
        apply_lora(self.model, rank=lora_rank, alpha=lora_alpha)
        saved = dict(mx.load(trainable))
        self.model.update(tree_unflatten(list(saved.items())))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.calls = 0

    def predict(self, payload, batch_questions=0, temperature=1.0):
        from predict_mlx_decisions import HELPERS
        t0 = time.perf_counter()
        states = HELPERS.validate_request(payload)
        examples = HELPERS.prepare_examples(payload, self.tokenizer, self.limit)
        batches = HELPERS.complete_question_batches(examples, batch_questions)
        self.calls += 1
        outputs = {s["id"]: {"id": s["id"], "answers": {}} for s in states}
        import numpy as _np
        for batch in batches:
            logits, _ = self.model(batch, self.tokenizer.pad_token_id)
            mx.eval(logits)
            ln = _np.array(logits, dtype=_np.float64)
            for example, values in zip(batch, ln):
                k = len(example["candidate_ids"])
                scores = values[:k] / temperature
                scores -= scores.max()
                exp = _np.exp(scores)
                probs = (exp / exp.sum()).tolist()
                outputs[example["state_id"]]["answers"][example["qid"]] = \
                    HELPERS.answer_from_probabilities(example, probs)
        n_paths = sum(len(ex["leaf_tokens"]) for ex in examples)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "execution": {"device": "mlx", "forward_passes": len(batches),
                          "network_model_calls": 0,
                          "server_evaluation_seconds": time.perf_counter() - t0},
            "states": list(outputs.values()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--trainable", required=True)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--splits", default="test,ood")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-states", type=int, default=2)
    p.add_argument("--batch-questions", type=int, default=0)
    p.add_argument("--max-length", type=int, default=2048)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a new output file")
    source = a.episodes.read_bytes()
    splits = {v.strip() for v in a.splits.split(",") if v.strip()}
    selected = [row for row in map(json.loads, filter(str.strip, source.decode().splitlines()))
                if row["game"] == "scaled_maze" and row["split"] in splits]
    if not selected:
        raise ValueError("No selected maze episodes")
    engine = MLXEdgesEngine(a.checkpoint_dir, a.trainable, max_length=a.max_length,
                            lora_rank=a.lora_rank, lora_alpha=a.lora_alpha)
    result = run_exploration(selected, engine, 5, a.max_steps, a.batch_states, a.batch_questions)
    result.update(model={"engine": "mlx", "trainable": str(a.trainable)},
                  selected_episode_ids=[row["id"] for row in selected],
                  source_episodes_sha256=hashlib.sha256(source).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), "summary": result["summary"],
                      "execution": result["execution"]}), flush=True)


if __name__ == "__main__":
    main()
