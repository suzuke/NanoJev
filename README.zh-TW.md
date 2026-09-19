# NanoJev — MLX 移植與 M3 訓練分支（繁體中文）

**[English](README.en.md)** | [简体中文](README.zh-CN.md) | **繁體中文**

> 上游專案：[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) —— Jev 的 nano 級開源復刻：
> 平行決策、動態候選、端到端訓練管線。本分支的所有工作都在 Apple Silicon（M3 Max）上完成。

## 這個分支做了什麼

1. **MPS 推理**（`scripts/predict_mps_decisions.py`）：解開原版寫死的 CUDA 限制，M3 可直接跑真模型推理。
2. **MLX 移植**（`scripts/mlx_decision_model.py`、`scripts/predict_mlx_decisions.py`）：
   Qwen3 backbone + 決策頭完整搬到 MLX，與 PyTorch 版逐數對齊（全域最大差 6.4e-07，FP32 舍入等級）。
   推理 7 paths / 1 forward 約 0.1 秒。
3. **MLX 訓練**：`scripts/train_mlx_heads.py`（凍 backbone，只訓頭，300 步約 23 秒）、
   `scripts/train_mlx_lora.py`（backbone 掛 rank-8 LoRA 聯訓）、`scripts/evaluate_mlx_heads.py`（評測）。
4. **實驗記錄**（`runs_mlx/EXPERIMENT_RECORD.md`）：官方數據對照、復現指令、速度分析。

## 成績

| 任務 | released 起點 | 本分支 | 官方全量微調 |
|---|---|---|---|
| maze local（test / 50×50 ood） | 56.25% / 56.25% | **93.75% / 92.19%**（LoRA+heads） | 77.84% / 76.56% |
| policy snake（test / ood） | 59.3% / 57.2% | **69.2% / 65.3%**（LoRA 早停） | —（官方只報閉環） |

maze 用 135 萬可訓練參數、dev-NLL 選 step 25 就超過官方全量微調。
权重在 `runs_mlx/local_lora3/trainable_best.safetensors`（maze）與
`runs_mlx/policy_lora/best_s6.safetensors`（snake）。

## 快速開始（M3 Mac）

```bash
git clone -b mlx-m3-port https://github.com/suzuke/NanoJev.git
cd NanoJev
uv venv .venv-mlx --python 3.12
uv pip install --python .venv-mlx/bin/python mlx mlx-lm huggingface_hub numpy transformers safetensors
```

下載 released checkpoint（約 2.2GB）：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"])
```

MLX 推理：

```bash
.venv-mlx/bin/python scripts/predict_mlx_decisions.py \
  --checkpoint-dir checkpoints/NanoJev --input request.json --output pred.json
```

maze LoRA 訓練（官方 `games_v4` 資料，約 40 分鐘 / 50 steps）：

```bash
.venv-mlx/bin/python scripts/train_mlx_lora.py \
  --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
  --output-dir runs_mlx/local_lora3 --steps 50 --batch-questions 16 \
  --lora-rank 8 --lr 1e-4 --eval-every 25 --save-every 25 --max-length 2048 --seed 17
```

## 已知的坑（都踩過了）

- Qwen3 是 decoder-only：PyTorch 端全 1 mask 實際走 causal attention，MLX 這邊也要補因果遮罩，否則差 0.37。
- autograd 不能經過 numpy：pack / set-attention 的 index-add 都要寫成純 mx 算子，
  否則 LoRA 的 B 矩陣梯度恆零、训练只動到 heads。
- 精度都量過：FP16、4-bit 量化、`mx.compile` 在 0.6B + 小 batch 上全部比 FP32 dense 慢。FP32 留著。
- snake 全量 300 步在 M3 上約 25 小時：不是 code 問題，是 1531 題 × 每步全 backbone 反向的物理量。已用早停（dev 第 5 步攻頂）處理。

## 授權

沿用上游 MIT License。模型與資料來自 [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev)
與 [C-Tianyu/NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)。
