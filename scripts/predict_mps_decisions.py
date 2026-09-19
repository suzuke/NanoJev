#!/usr/bin/env python3
"""M3/MPS/CPU 推理入口：复用 predict_toy_decisions 的 schema/编码，只放宽设备限制。

原版 DecisionPredictor 写死 cuda:0 + torch.autocast("cuda") + BF16 检查，
在 Apple Silicon 上会直接 raise。本文件不改原文件、不改权重，
只做：
  - 允许 --device mps / cpu (/ cuda 仍走原逻辑)
  - mps/cpu 默认 fp32（关闭 autocast，数值最稳）
  - attn sdpa 失败时回退 eager
输出 schema 与原版一致，多一个 execution.mps_patch 标记。
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

from predict_toy_decisions import (
    answer_from_probabilities,
    complete_question_batches,
    load_decision_model_class,
    local_checkpoint_files,
    prepare_examples,
    read_json,
    validate_request,
)


class MPSDecisionPredictor:
    """与 DecisionPredictor 同接口，device-agnostic。"""

    def __init__(self, checkpoint_dir, max_length=None, device_name="mps",
                 disable_native_triton=False, precision="fp32"):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision 必须为 fp32 或 bf16")
        root, paths = local_checkpoint_files(checkpoint_dir)
        run_config = read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config 缺少合法 set_head")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if disable_native_triton:
            from torch._native import triton_utils
            triton_utils.deregister_op_overrides()
        device = torch.device(device_name)
        effective_precision = precision
        precision_note = ""
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("cuda 不可用，M3 上请用 --device mps 或 cpu")
            torch.cuda.set_device(device)
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("当前CUDA设备不支持BF16")
            torch.backends.cuda.matmul.allow_tf32 = False
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("mps 不可用，请改用 --device cpu")
            if precision == "bf16":
                # MPS 的 bfloat16 autocast 不稳，直接跑 fp32 并记录。
                effective_precision = "fp32"
                precision_note = "mps不支持bf16 autocast，已回退fp32"
        elif device.type == "cpu":
            if precision == "bf16":
                effective_precision = "fp32"
                precision_note = "cpu上bf16 autocast已关闭，改用fp32"
        else:
            raise ValueError(f"不支持的 device: {device_name}（可用 cuda/mps/cpu）")

        tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                 trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True,
                                                trust_remote_code=False)
        body_config.use_cache = False
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        if type(limit) is not int or limit <= 0:
            raise ValueError("max-length 必须为正整数")
        context_limit = getattr(body_config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and limit > context_limit:
            raise ValueError("max-length 超过backbone配置声明的上下文长度")

        try:
            body = AutoModel.from_config(body_config, attn_implementation="sdpa",
                                        trust_remote_code=False).float()
        except Exception:
            body = AutoModel.from_config(body_config, trust_remote_code=False).float()
            precision_note += ("; sdpa attention在当前设备失败，已回退eager" if precision_note
                               else "sdpa attention在当前设备失败，已回退eager")
        DecisionModel = load_decision_model_class()
        model = DecisionModel(body, run_config["set_head"])
        weights = load_file(str(paths["weights"]), device="cpu")
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(device=device, dtype=torch.float32)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.device = device
        self.precision = precision
        self.effective_precision = effective_precision
        self.precision_note = precision_note
        self.disable_native_triton = disable_native_triton
        self.inference_calls = 0
        self._torch = torch

    def predict(self, payload, batch_questions=0, temperature=1.0):
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
                or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        torch = self._torch
        model, tokenizer = self.model, self.tokenizer
        root, run_config, limit = self.root, self.run_config, self.limit
        device = self.device
        use_autocast = (self.effective_precision == "bf16" and device.type == "cuda")
        examples = prepare_examples(payload, tokenizer, limit)
        batches = complete_question_batches(examples, batch_questions)
        self.inference_calls += 1
        model.eval()
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        with torch.inference_mode():
            for batch in batches:
                if use_autocast:
                    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)
                else:
                    import contextlib
                    ctx = contextlib.nullcontext()
                with ctx:
                    logits, _ = model(batch, tokenizer.pad_token_id)
                for example, values in zip(batch, logits):
                    k = len(example["candidate_ids"])
                    scores = values[:k].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("模型产生非有限logits，未返回部分预测")
                    probabilities = (scores / temperature).softmax(-1).cpu().tolist()
                    outputs[example["state_id"]]["answers"][example["qid"]] = \
                        answer_from_probabilities(example, probabilities)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(root), "base_model": run_config.get("model"),
                           "base_revision": run_config.get("resolved_model_revision"),
                           "set_head": run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": str(device), "parameter_storage": "float32",
                          "precision": self.effective_precision,
                          "precision_requested": self.precision,
                          "precision_note": self.precision_note,
                          "forward_autocast": "bfloat16" if use_autocast else "disabled",
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                          "forward_passes": len(batches),
                          "batch_questions_limit": batch_questions or "all",
                          "autoregressive_decode_steps": 0, "prefix_sharing": False,
                          "max_length": limit, "disable_native_triton": self.disable_native_triton,
                          "network_model_calls": 0, "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls,
                          "mps_patch": True},
            "states": list(outputs.values()),
        }


def predict(payload, checkpoint_dir, temperature=1.0, batch_questions=0, max_length=None,
            device_name="mps", disable_native_triton=False, precision="fp32"):
    validate_request(payload)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数")
    engine = MPSDecisionPredictor(checkpoint_dir, max_length=max_length, device_name=device_name,
                                  disable_native_triton=disable_native_triton, precision=precision)
    return engine.predict(payload, batch_questions=batch_questions, temperature=temperature)


def main():
    import torch
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="含states数组的JSON文件")
    parser.add_argument("--output", help="不设置时将完整结果输出到stdout")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=default_device, help="mps / cpu / cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--disable-native-triton", action="store_true")
    args = parser.parse_args()
    try:
        result = predict(read_json(args.input), args.checkpoint_dir, args.temperature,
                         args.batch_questions, args.max_length, args.device,
                         args.disable_native_triton, precision=args.precision)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(destination), "execution": result["execution"]},
                             ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
