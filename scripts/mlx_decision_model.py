#!/usr/bin/env python3
"""MLX 版 DecisionModel：与 scripts/train_toy_decisions.py 的 DecisionModel 数学等价。

结构：
  backbone: mlx-lm Qwen3Model（28层 GQA，与 Qwen3-0.6B 配置一致）
  heads: norm(LayerNorm 1024) + scalar(Linear 1024->1)
         + set attention（与 PyTorch nn.MultiheadAttention(128,4) 等价，
           用 4 个 nn.Linear 实现以便训练时求导）
         + set_project(Linear 1025->128) + set_output(Linear 128->1)

权重来源：NanoJev best.safetensors（backbone.* 是微调后的 Qwen3Model 权重，
直接搬运，非官方 stock 权重）。
"""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

SET_DIMS = 128
SET_HEADS = 4


class LoRALinear(nn.Module):
    """base Linear 冻结 + 低秩适配。forward 数学：y = base(x) + scale * B(A(x))。"""

    def __init__(self, in_dims: int, out_dims: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = nn.Linear(in_dims, out_dims, bias=False)
        self.lora_a = nn.Linear(in_dims, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dims, bias=False)
        self.scale = alpha / rank
        self.rank = rank

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale

    def fused_weight(self) -> mx.array:
        b = _np_for_fuse(self.lora_b.weight)
        a = _np_for_fuse(self.lora_a.weight)
        import numpy as _np
        delta = (b.T @ a.T).T * self.scale  # (out,in)
        return self.base.weight + mx.array(delta)


def _np_for_fuse(a: mx.array):
    import numpy as _np
    mx.eval(a)
    return _np.array(a)


def apply_lora(model, rank: int = 8, alpha: float = 16.0,
               targets=("q_proj", "v_proj"), seed: int = 17):
    """把 backbone 每层 self_attn 的目标投影换成 LoRALinear（base 权重原位保留）。

    返回注入的 LoRALinear 列表。调用后请 freeze 全模型再 unfreeze 训练目标。
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    injected = []
    for li, layer in enumerate(model.backbone.layers):
        attn = layer.self_attn
        for name in targets:
            base = getattr(attn, name)
            in_d = base.weight.shape[1]
            out_d = base.weight.shape[0]
            lora = LoRALinear(in_d, out_d, rank=rank, alpha=alpha)
            lora.base.weight = base.weight
            bound = 1.0 / math.sqrt(in_d)
            lora.lora_a.weight = mx.array(
                rng.uniform(-bound, bound, (rank, in_d)).astype(_np.float32))
            lora.lora_b.weight = mx.zeros((out_d, rank))
            setattr(attn, name, lora)
            injected.append(((li, name), lora))
    mx.eval(model.parameters())
    return injected


def freeze_for_lora_heads(model):
    """冻结全模型，只解冻 heads + 全部 LoRA A/B。返回可训练参数量。"""
    model.freeze()
    model.heads.unfreeze()
    for layer in model.backbone.layers:
        for proj in ("q_proj", "v_proj", "k_proj", "o_proj"):
            mod = getattr(layer.self_attn, proj, None)
            if isinstance(mod, LoRALinear):
                mod.lora_a.unfreeze()
                mod.lora_b.unfreeze()
    import numpy as _np
    n = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        model.trainable_parameters()))
    return int(n)


def build_args(backbone_cfg: dict) -> ModelArgs:
    return ModelArgs(
        model_type=backbone_cfg.get("model_type", "qwen3"),
        hidden_size=backbone_cfg["hidden_size"],
        num_hidden_layers=backbone_cfg["num_hidden_layers"],
        intermediate_size=backbone_cfg["intermediate_size"],
        num_attention_heads=backbone_cfg["num_attention_heads"],
        rms_norm_eps=backbone_cfg.get("rms_norm_eps", 1e-6),
        vocab_size=backbone_cfg["vocab_size"],
        num_key_value_heads=backbone_cfg["num_key_value_heads"],
        max_position_embeddings=backbone_cfg.get("max_position_embeddings", 40960),
        rope_theta=backbone_cfg.get("rope_parameters", {}).get("rope_theta", 1000000.0),
        head_dim=backbone_cfg["head_dim"],
        tie_word_embeddings=backbone_cfg.get("tie_word_embeddings", True),
        rope_scaling=None,
    )


class MLXHeads(nn.Module):
    """可训练的决策头（backbone 冻结时只优化这部分）。"""

    def __init__(self, hidden: int, set_head: str = "attention"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)          # eps 1e-5，同 PyTorch 默认
        self.scalar = nn.Linear(hidden, 1)
        self.set_head = set_head
        if set_head == "attention":
            self.set_project = nn.Linear(hidden + 1, SET_DIMS)
            self.set_q_proj = nn.Linear(SET_DIMS, SET_DIMS)
            self.set_k_proj = nn.Linear(SET_DIMS, SET_DIMS)
            self.set_v_proj = nn.Linear(SET_DIMS, SET_DIMS)
            self.set_o_proj = nn.Linear(SET_DIMS, SET_DIMS)
        self.set_output = nn.Linear(SET_DIMS, 1)

    def _set_attention(self, u: mx.array, valid: mx.array) -> mx.array:
        """等价于 PyTorch nn.MultiheadAttention(128,4,batch_first,dropout=0) eval 行为。

        u: (B,K,128)，valid: (B,K) bool。返回 mixed: (B,K,128)。
        """
        B, K, D = u.shape
        H, Dh = SET_HEADS, D // SET_HEADS
        q = self.set_q_proj(u)
        k = self.set_k_proj(u)
        v = self.set_v_proj(u)
        q = q.reshape(B, K, H, Dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, K, H, Dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, K, H, Dh).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
        # key_padding_mask=True 表示忽略 -> additive -inf
        add = mx.where(valid[:, None, None, :], 0.0, float("-inf"))
        attn = mx.softmax(scores + add, axis=-1)
        mixed = (attn @ v).transpose(0, 2, 1, 3).reshape(B, K, D)
        return self.set_o_proj(mixed)

    def __call__(self, h: mx.array, valid_q: mx.array, types: list) -> mx.array:
        """h: (Q,Kmax,H) packed leaves；valid_q: (Q,Kmax) bool；types: 每题类型。

        全程 mx 算子（pad/stack/gather），自动微分可穿透到 backbone/LoRA。
        """
        kmax = h.shape[1]
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1)  # (Q,K)
        delta = None
        if self.set_head == "attention":
            choice_idx = [i for i, t in enumerate(types) if t == "choice"]
            if choice_idx:
                idx = mx.array(choice_idx)
                hc = h[idx]
                vc = valid_q[idx]
                log_k = mx.log(vc.sum(axis=-1, keepdims=True).astype(mx.float32))
                log_k = mx.broadcast_to(log_k[:, :, None], (hc.shape[0], kmax, 1))
                u = self.set_project(mx.concatenate([hc, log_k], axis=-1))
                mixed = self._set_attention(u, vc)
                delta = self.set_output(mx.tanh(u + mixed)).squeeze(-1)  # (C,K)
        rows = []
        ci = 0
        for i, t in enumerate(types):
            if t == "boolean":
                row = mx.concatenate([z[i, 0:1] * 0, z[i, 0:1]], axis=0)
                if kmax > 2:
                    row = mx.concatenate(
                        [row, mx.full((kmax - 2,), float("-inf"), dtype=row.dtype)], axis=0)
                rows.append(row)
            else:
                r = z[i]
                if t == "choice" and delta is not None:
                    r = r + delta[ci]  # gather，可微
                    ci += 1
                rows.append(r)
        logits = mx.stack(rows)
        return mx.where(valid_q, logits, float("-inf"))


def pack_leaves_mx(leaves: mx.array, n_list: list, k_list: list):
    """leaves: (P,H) → (h:(Q,Kmax,H), valid:(Q,Kmax)bool)。纯 mx，可微分。"""
    segs, off = [], 0
    for n in n_list:
        segs.append(leaves[off : off + n])
        off += n
    return pack_segments_mx(segs, k_list)


def pack_segments_mx(segs: list, k_list: list):
    """segs: 每题的 (n,H) mx 切片 → (h, valid)。纯 mx，可微分。"""
    kmax = max(k_list)
    rows, masks = [], []
    for seg, k in zip(segs, k_list):
        n = seg.shape[0]
        if n < kmax:
            seg = mx.pad(seg, [(0, kmax - n), (0, 0)])
        rows.append(seg)
        masks.append(mx.array([True] * k + [False] * (kmax - k)))
    return mx.stack(rows), mx.stack(masks)


class MLXDecisionModel(nn.Module):
    def __init__(self, backbone_cfg: dict, set_head: str = "attention"):
        super().__init__()
        self.backbone = Qwen3Model(build_args(backbone_cfg))
        self.heads = MLXHeads(backbone_cfg["hidden_size"], set_head)

    def forward_paths(self, tokens: mx.array, valid_mask: mx.array) -> mx.array:
        """tokens: (P,L) uint32；valid_mask: (P,L) bool。返回每条 path 取末有效位的 hidden (P,H)。

        与 PyTorch 端一致：decoder-only 因果 attention + padding 遮罩。
        右侧 padding 只出现在末尾，query 侧无需遮（读取位置恒为有效位）。
        """
        h = self.backbone.embed_tokens(tokens)
        P, L = tokens.shape
        causal = mx.tril(mx.ones((L, L), dtype=mx.bool_))
        key_ok = valid_mask[:, None, None, :] & causal[None, None, :, :]
        add = mx.where(key_ok, 0.0, float("-inf")).astype(h.dtype)
        for layer in self.backbone.layers:
            h = layer(h, add)
        h = self.backbone.norm(h)
        lengths = valid_mask.sum(axis=-1)  # (P,)
        idx = (lengths - 1).astype(mx.uint32)
        return h[mx.arange(h.shape[0]), idx]

    def pack_examples(self, examples, pad_id: int):
        """examples -> (h_packed, valid_q, types)。h_packed 仍需 backbone leaves，见 train 脚本缓存流程。"""
        paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
        lengths = [len(ids) for ids in paths]
        width = max(lengths)
        tok = [[ids[i] if i < len(ids) else pad_id for i in range(width)] for ids in paths]
        tokens = mx.array(tok, dtype=mx.uint32)
        valid = mx.array([[i < n for i in range(width)] for n in lengths])
        return tokens, valid

    def __call__(self, examples, pad_id: int):
        tokens, valid = self.pack_examples(examples, pad_id)
        leaves = self.forward_paths(tokens, valid)  # (P,H)
        n_list = [len(ex["leaf_tokens"]) for ex in examples]
        k_list = [len(ex["candidate_ids"]) for ex in examples]
        h, valid_q = pack_leaves_mx(leaves, n_list, k_list)
        logits = self.heads(h, valid_q, [ex["type"] for ex in examples])
        return logits, valid_q


def mx_to_np(a: mx.array):
    import numpy as _np
    mx.eval(a)
    return _np.array(a, copy=False)


def _unflatten(flat: dict) -> dict:
    """把 backbone.layers.0.self_attn.q_proj.weight 轉成巢狀 dict/list（mlx update 格式）。"""
    root: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        node = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            nxt = parts[i + 1] if not last else None
            if part.isdigit():
                idx = int(part)
                while len(node) <= idx:
                    node.append(None)
                if last:
                    node[idx] = val
                else:
                    if node[idx] is None:
                        node[idx] = [] if nxt.isdigit() else {}
                    node = node[idx]
            else:
                if last:
                    node[part] = val
                else:
                    if part not in node:
                        node[part] = [] if nxt.isdigit() else {}
                    node = node[part]
    return root


HEAD_KEYS = ("norm.weight", "norm.bias", "scalar.weight", "scalar.bias",
             "set_project.weight", "set_project.bias",
             "set_output.weight", "set_output.bias")


def split_checkpoint_weights(sft_path: str):
    """读 best.safetensors，返回 (backbone_flat, heads_flat, set_attn_splits)。"""
    from safetensors.numpy import load_file
    raw = load_file(sft_path)
    backbone_flat, heads_flat = {}, {}
    in_w = in_b = out_w = out_b = None
    for k, v in raw.items():
        if k.startswith("backbone."):
            backbone_flat[k[len("backbone.") :]] = mx.array(v)
        elif k in HEAD_KEYS:
            heads_flat[k] = mx.array(v)
        elif k == "set_attention.in_proj_weight":
            in_w = v
        elif k == "set_attention.in_proj_bias":
            in_b = v
        elif k == "set_attention.out_proj.weight":
            out_w = v
        elif k == "set_attention.out_proj.bias":
            out_b = v
        else:
            raise ValueError(f"未知权重键：{k}")
    splits = {"in_w": mx.array(in_w), "in_b": mx.array(in_b),
              "out_w": mx.array(out_w), "out_b": mx.array(out_b)}
    return backbone_flat, heads_flat, splits


def apply_set_splits(heads: MLXHeads, splits: dict):
    """把 PyTorch MHA 打包权重拆进 4 个 Linear（Q/K/V 各 128，出投影 128）。"""
    heads.set_q_proj.weight = splits["in_w"][0:128]
    heads.set_k_proj.weight = splits["in_w"][128:256]
    heads.set_v_proj.weight = splits["in_w"][256:384]
    heads.set_q_proj.bias = splits["in_b"][0:128]
    heads.set_k_proj.bias = splits["in_b"][128:256]
    heads.set_v_proj.bias = splits["in_b"][256:384]
    heads.set_o_proj.weight = splits["out_w"]
    heads.set_o_proj.bias = splits["out_b"]


def load_heads_from_checkpoint(heads: MLXHeads, sft_path: str):
    """只搬运决策头（warm-start，backbone 另行处理）。"""
    _backbone, heads_flat, splits = split_checkpoint_weights(sft_path)
    heads.update(_unflatten(heads_flat))
    apply_set_splits(heads, splits)
    mx.eval(heads.parameters())
    return heads


def load_weights(model: MLXDecisionModel, sft_path: str):
    """从 best.safetensors 搬运全部权重（numpy 中转，128GB 统一内存无压力）。"""
    backbone_flat, heads_flat, splits = split_checkpoint_weights(sft_path)
    model.backbone.update(_unflatten(backbone_flat))
    model.heads.update(_unflatten(heads_flat))
    apply_set_splits(model.heads, splits)
    mx.eval(model.parameters())
    return model


def init_heads_random(heads: MLXHeads, hidden: int, seed: int = 17):
    """与 PyTorch 端一致的随机初始化：
    scalar ~ N(0,0.02)；set_output 全零；set_project Uniform(±1/sqrt(fan_in))；
    set attention 按 MHA 默认（in Xavier uniform 全矩阵后拆，bias 零；out Xavier uniform，bias 零）。
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    heads.norm.weight = mx.ones((hidden,))
    heads.norm.bias = mx.zeros((hidden,))
    heads.scalar.weight = mx.array(rng.normal(0, 0.02, (1, hidden)).astype(_np.float32))
    heads.scalar.bias = mx.zeros((1,))
    if heads.set_head == "attention":
        b = 1.0 / math.sqrt(hidden + 1)
        heads.set_project.weight = mx.array(
            rng.uniform(-b, b, (SET_DIMS, hidden + 1)).astype(_np.float32))
        heads.set_project.bias = mx.array(rng.uniform(-b, b, (SET_DIMS,)).astype(_np.float32))
    heads.set_output.weight = mx.zeros((1, SET_DIMS))
    heads.set_output.bias = mx.zeros((1,))
    if heads.set_head == "attention":
        xb = math.sqrt(6.0 / (SET_DIMS + 3 * SET_DIMS))
        full_w = rng.uniform(-xb, xb, (3 * SET_DIMS, SET_DIMS)).astype(_np.float32)
        heads.set_q_proj.weight = mx.array(full_w[0:128])
        heads.set_k_proj.weight = mx.array(full_w[128:256])
        heads.set_v_proj.weight = mx.array(full_w[256:384])
        for p in (heads.set_q_proj, heads.set_k_proj, heads.set_v_proj):
            p.bias = mx.zeros((SET_DIMS,))
        ob = math.sqrt(6.0 / (SET_DIMS + SET_DIMS))
        heads.set_o_proj.weight = mx.array(
            rng.uniform(-ob, ob, (SET_DIMS, SET_DIMS)).astype(_np.float32))
        heads.set_o_proj.bias = mx.zeros((SET_DIMS,))
    mx.eval(heads.parameters())
    return heads
