#!/usr/bin/env python3
"""MLX 推理入口：与 predict_mps_decisions.py 同输入输出 schema，跑在 Apple Silicon 原生后端。

用法：
  .venv-mlx/bin/python scripts/predict_mlx_decisions.py \
    --checkpoint-dir checkpoints/NanoJev \
    --input /tmp/nanojev_request.json --output /tmp/nanojev_pred_mlx.json
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_decision_model import MLXDecisionModel, load_weights, mx_to_np


def load_toy_helpers():
    path = Path(__file__).with_name("predict_toy_decisions.py")
    spec = importlib.util.spec_from_file_location("nanojev_toy_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HELPERS = load_toy_helpers()


class MLXDecisionPredictor:
    def __init__(self, checkpoint_dir, max_length=None):
        root = Path(checkpoint_dir).expanduser().resolve(strict=True)
        run_config = json.loads((root / "config.json").read_text())
        if run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config 缺少合法 set_head")
        backbone_cfg = json.loads((root / "backbone_config" / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"),
                                                 local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if type(tokenizer.eos_token_id) is not int:
            raise ValueError("checkpoint tokenizer 必须有合法 eos_token_id")
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        self.model = MLXDecisionModel(backbone_cfg, run_config["set_head"])
        load_weights(self.model, str(root / "best.safetensors"))
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.inference_calls = 0

    def predict(self, payload, batch_questions=0, temperature=1.0):
        import time
        t0 = time.perf_counter()
        states = HELPERS.validate_request(payload)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        examples = HELPERS.prepare_examples(payload, self.tokenizer, self.limit)
        batches = HELPERS.complete_question_batches(examples, batch_questions)
        self.inference_calls += 1
        self.model.eval()
        outputs = {s["id"]: {"id": s["id"], "answers": {}} for s in states}
        import numpy as _np
        for batch in batches:
            logits, _ = self.model(batch, self.tokenizer.pad_token_id)
            mx.eval(logits)
            ln = _np.array(logits, dtype=_np.float64)
            for example, values in zip(batch, ln):
                k = len(example["candidate_ids"])
                scores = values[:k] / temperature
                scores = scores - _np.max(scores)
                exp = _np.exp(scores)
                probs = (exp / exp.sum()).tolist()
                outputs[example["state_id"]]["answers"][example["qid"]] = \
                    HELPERS.answer_from_probabilities(example, probs)
        dt = time.perf_counter() - t0
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(self.root),
                           "base_model": self.run_config.get("model"),
                           "base_revision": self.run_config.get("resolved_model_revision"),
                           "set_head": self.run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": "mlx", "parameter_storage": "float32",
                          "precision": "float32", "forward_autocast": "disabled",
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                          "forward_passes": len(batches),
                          "batch_questions_limit": batch_questions or "all",
                          "autoregressive_decode_steps": 0, "prefix_sharing": False,
                          "max_length": self.limit,
                          "network_model_calls": 0, "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls,
                          "mlx_patch": True, "predict_seconds": dt},
            "states": list(outputs.values()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-questions", type=int, default=0)
    p.add_argument("--max-length", type=int, default=None)
    a = p.parse_args()
    try:
        eng = MLXDecisionPredictor(a.checkpoint_dir, max_length=a.max_length)
        payload = json.loads(Path(a.input).read_text(encoding="utf-8"))
        result = eng.predict(payload, batch_questions=a.batch_questions,
                             temperature=a.temperature)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(json.dumps({"output": a.output, "execution": result["execution"]},
                             ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
