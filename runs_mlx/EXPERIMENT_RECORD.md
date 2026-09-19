# MLX on M3 Max — NanoJev 复现实验记录

> 环境：MacBook Pro M3 Max（16 核，128GB 统一内存），macOS arm64
> 官方环境对照：`capyubara-0`，Python 3.14.4，NVIDIA A100-SXM4-80GB，torch 2.14，BF16

## 官方数据（TianyuCodings/NanoJev 公开）

### 训练配置

| 阶段 | steps | batch | max-length | 数据量 |
|---|---|---|---|---|
| base run | 600 + 12 head-only | 12 questions | 512 | 2,312 states / 6,936 questions |
| navigation 续训 | 1200 | 12 | 512 | 500 maps / 3,000 states / 9,000 questions |
| local maze（展示用） | 300 | 16 | 2048 | 300 states / 1,200 questions |
| policy snake（展示用） | 300 | 12 | 8192 | 589 states / 3,195 questions |
| event 校准 ×3 臂 | 300 | 16 | 8192 | train 1,124 / dev 372 / test 364 / ood 128 |

全量：backbone Qwen3-0.6B（revision `c1899de`）+ 全参数 AdamW（backbone-lr 2e-5 / head-lr 2e-4），
BF16 autocast，dev target cross-entropy 选最优。

### 官方成绩

- local safety：test **77.84%** / 50×50 ood **76.56%**（scalar Brier 0.14358 / 0.15599，NLL 0.44045 / 0.44656）
- paired proper-reward：test L2 0.11844 / ood 0.06202
- 50×50 maze 闭环：244 attempts / 36 碰撞 / 到达
- 12×12 snake 闭环：27 食物 / 256 步 / 存活到 horizon
- 早期 40-map 导航：test 19/20，ood 18/20

## 本机（M3 Max + MLX）结果

### 1. 推理移植：数值对齐 ✅

- `scripts/mlx_decision_model.py` + `scripts/predict_mlx_decisions.py`
- Qwen3 backbone（28 层 GQA）+ norm/scalar/set-attention heads，权重逐张量搬运（抽查 maxdiff = 0.0）
- 同一份 request，PyTorch MPS 版 vs MLX 版：**global max_abs_diff = 6.4e-07**（fp32 舍入级）
- 速度：7 paths / 1 forward，MLX predict 0.10s（含载权重整趟约 3s，MPS 版 18s）
- 关键坑：Qwen3 是 decoder-only，PyTorch 端全 1 mask 实际走 causal attention，初版双向实现差 0.37，补因果遮罩后对齐

### 2. Maze local atomic：MLX LoRA+heads ✅（超官方）

- 配置：backbone 冻结 + q/v LoRA（rank 8，alpha 16，~0.9M）+ heads 全训（~0.4M），合计 1,347,458 可训练参数；
  AdamW lr 1e-4，300 步计划，dev-NLL 选优（`scripts/train_mlx_lora.py`）
- 修过的真 bug：初版 pack 经 numpy 中转切断 autograd，LoRA B 梯度恒零；
  改为纯 mx（pad/stack/gather）后梯度正常
- 结果（`runs_mlx/local_lora3/trainable_best.safetensors`，dev-NLL 选 step 25）：

| | test（176 题） | ood 50×50（64 题） |
|---|---|---|
| released 起点 | 56.25%（=多数类基线） | 56.25% |
| heads-only 300 步 | 57.4%（Brier 0.237） | 57.8% |
| **LoRA+heads（本机）** | **93.75%（Brier 0.049，NLL 0.223）** | **92.19%（Brier 0.067，NLL 0.362）** |
| 官方全量微调 | 77.84%（Brier 0.144，NLL 0.440） | 76.56%（Brier 0.156，NLL 0.447） |

- 耗时：约 36 分钟 / 50 steps（含 dev 评测）；训练 loss 快速归零（过拟合），dev 25 步攻顶后回升——早停是关键
- 解读：小数据（576 train）上冻结 backbone + 小适配器比全量微调泛化更好；评测见 `scripts/evaluate_mlx_heads.py`

### 3. Policy（snake+maze）：heads-only ✅，LoRA ✅（早停 5 步）

- heads-only 300 步（23 秒级）：test 59.3%→**63.7%**，ood 57.2%→**64.7%**，Brier 0.172→0.111
  （`runs_mlx/scaled_policy/heads_best.safetensors`）
- LoRA 联训（同 maze 配置，lr 1e-4）：dev-NLL 首评（step 5）即攻顶 0.608，
  之后迅速过拟合（step 11 dev 0.77；降 lr 到 5e-5 也救不回）。
  取 step-5 权重（`runs_mlx/policy_lora/best_s6.safetensors`）：

| | test（487 题） | ood（173 题） |
|---|---|---|
| released 起点 | 59.3%（NLL 0.86） | 57.2%（NLL 0.83） |
| heads-only | 63.7%（NLL 0.71） | 64.7%（NLL 0.63） |
| **LoRA 早停 step-5** | **69.2%（NLL 0.58，Brier 0.110）** | **65.3%（NLL 0.61，Brier 0.090）** |

- 耗时：约 40 分钟/6 steps；全量 300 步在 M3 上约 25 小时，不划算故早停（见下节速度分析）
- 闭环评测（`evaluate_model_edges_maze.py` 等）仍需移植 engine，未做

### 4. 文件索引

- 环境：`NanoJev/.venv`（torch 2.14 推理/数据生成），`NanoJev/.venv-mlx`（mlx 0.32.2 + mlx-lm，Python 3.12）
- 数据：`data/NanoJev/games_v4/{local_maze_v1,scaled_games_v4b/policy}`（HF 下载，与官方 manifest 一致，已 validate-only）
- 本机新增脚本：`predict_mps_decisions.py`，`mlx_decision_model.py`，`predict_mlx_decisions.py`，
  `train_mlx_heads.py`，`train_mlx_lora.py`，`evaluate_mlx_heads.py`
- 产物：`runs_mlx/heads_smoke`，`runs_mlx/local_atomic`，`runs_mlx/scaled_policy`，
  `runs_mlx/local_lora*`，`runs_mlx/policy_lora*`（后者为空/未完成）

## 附：精度实验（是否一定得 FP32？）

三種全實測過（maze，16 題一批，前向+反向）：

| 精度 | 時間 | 備註 |
|---|---|---|
| FP32 dense | 0.73s | 最快，留用 |
| FP16 dense | 前向 0.71s（FP32 0.31s） | 更慢，logits 差 2e-3 |
| 4-bit 量化 backbone（QLoRA 式，梯度可通，loss 0.818 vs 0.847） | 0.81s | 更慢，反量化開銷 > 流量節省 |

结论：0.6B + 小 batch 在 M3 上是 kernel-launch 开销主导，不是显存带宽主导，
量化省下的流量补不回反量化计算。MLX 原生该吃的红利（Metal 后端、统一内存、懒求值）已经在吃了，
剩下的速度杠杆是步数（dev 25 步攻顶）、batch 大小和换 GPU，不是精度。FP32 留着。

### 4. 闭环验证：MLX 模型 + 官方 planner 跑真 maze ✅（3/3 到达）

- `scripts/evaluate_mlx_edges_maze.py`：复用官方 `run_exploration` + `EdgeExplorer`
  （边记忆、模型排序、已验证边 BFS 回位），engine 换成 MLX（local_lora3 best）。
  输入 `results/rollout_pilot_episodes.jsonl`，`--max-steps 0`（预算 2×size²）：

| episode | attempts | collisions | 结果 | 轨迹 atomic acc |
|---|---|---|---|---|
| maze test 8×8 | 16 | 0 | 到达 | 98.4% |
| maze test 16×16 | 29 | 1 | 到达 | 96.4% |
| maze ood 50×50 | 135 | 1 | 到达 | 96.0% |

- 对照官方 50×50 展示（244 attempts / 36 碰撞 / 到达）：ours 135 attempts / 1 碰撞。
  注意 pilot episode 与官方展示局未必同一种子，严格同局对比需同 seed；但 3/3、共 2 碰撞、
  轨迹 atomic 96.3% 已证明题面分数转化成了真实行走能力（`runs_mlx/edges_maze_local_lora3.json`）。
- 147 次 predict 调用，全 MLX 本地零 API 花费。

## 复现命令（maze 获胜版本）

```bash
cd NanoJev && source .venv-mlx/bin/activate
python scripts/train_mlx_lora.py \
  --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
  --output-dir runs_mlx/local_lora3 --steps 50 --batch-questions 16 \
  --lora-rank 8 --lr 1e-4 --eval-every 25 --save-every 25 --max-length 2048 --seed 17 \
  --init-trainable runs_mlx/local_lora/trainable_latest.safetensors
python scripts/evaluate_mlx_heads.py \
  --input data/NanoJev/games_v4/data/local_maze_v1 --checkpoint-dir checkpoints/NanoJev \
  --trainable runs_mlx/local_lora3/trainable_best.safetensors --max-length 2048
```
