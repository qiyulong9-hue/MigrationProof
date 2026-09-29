# 迁证智析 MigrationProof

Qwen3.5-0.8B 昇腾 NPU 模型迁移验收 Agent Skill

## 1. 环境版本

本项目全部结果均来自团队此前长期使用的 Shengteng 服务器。服务器上保留两组用途不同的冻结实验配置：

| 证据轨道 | 硬件 | CANN | PyTorch | torch_npu | FLA-NPU | 用途 |
|---|---|---|---|---|---|---|
| Shengteng 精度对齐 | 2×Ascend910B4-1 | 9.1.0 | 2.7.1 | 2.7.1.post8 | 26.6.0 | AscendC/Triton 100 步 Loss 对齐 |
| Shengteng 应用复跑 | 2×Ascend910B4-1 | 8.2.RC1 | 2.6.0+cpu | 2.6.0 | 未单独使用（MindSpeed-MM FlashAttention） | P2-C04R 续训、工程回归、三组性能对照 |

两组配置均属于 Shengteng 服务器历史项目成果，分别服务精度对齐与应用性能验证，不拼接为同一次运行。

## 2. 运行启动脚本

```bash
cd /path/to/migrationproof_qwen35_p2
source env/load_project_env.sh

# 工程回归
python -m pytest -q

# 任务配置校验
python -c "import json; from migrationproof.task_validation import validate_task; d=json.load(open('configs/official_alignment_task_20260918_v1.json')); print(validate_task(d))"

# P2-C04R 续训
bash scripts/run_p2_c04r_continuation.sh

# 独立性能对照（每组分别执行 baseline 与 optimized）
PAIR_ID=1 bash scripts/run_p2_c04r_benchmark_once.sh baseline
PAIR_ID=1 bash scripts/run_p2_c04r_benchmark_once.sh optimized

# 指标分析
python scripts/analyze_accuracy.py --help
python scripts/analyze_benchmark_pair_resumed.py --help
python scripts/analyze_independent_pairs.py --help

# 验收报告
python scripts/build_acceptance_report.py --help
```

运行前需按本机目录修改模型、数据、检查点和输出路径。每次运行必须使用新的输出目录，不覆盖历史日志。

## 3. 代码结构与逻辑

| 路径 | 作用 |
|---|---|
| `configs/` | 任务契约、训练配置与验收配置 |
| `data/` | MP-Bench 数据及 manifest |
| `env/` | 项目环境加载脚本 |
| `migrationproof/` | 精度、性能、诊断、验收与报告模块 |
| `scripts/` | 数据构建、训练、推理、评分和报告入口 |
| `tests/` | 工程回归与协议测试 |
| `evidence/` | 指标、证据账本和 SHA-256 |
| `SKILL.md` | Agent Skill 入口、安全边界和验收规则 |

代码主流程：任务校验 → 环境指纹 → 模型与数据身份 → 训练冒烟 → 精度门禁 → 性能公平性检查 → 三组独立对照 → 应用验收 → 证据账本与报告。

## 4. PR 链接

当前暂无可核验的上游 PR。提交材料中不虚构 PR 链接；后续仅在形成真实、可访问的最小兼容性修复后补充。

## 5. 项目简介

MigrationProof 用于验证 Qwen3.5-0.8B 迁移到 Huawei Ascend NPU 后是否正确、可复现且证据完整。Agent 负责任务理解、工具编排、异常解释和报告叙事；版本、哈希、Loss、时延、吞吐与门禁结论均由确定性程序计算。

## 6. 模型与数据

- 模型：Qwen3.5-0.8B。
- 训练框架：MindSpeed-MM / PyTorch FSDP。
- 精度对齐数据：COCO2017 与 LLaVA-Instruct-150K slim，冻结 2000 张图像。
- 主要精度配置：`cutoff_len=1024`、`seed=42`、`shuffle=false`、`global_batch_size=8`、`lr=1e-5`、100 steps。
- 应用数据：MigrationProof-Bench v0.6.1-p2，训练 360、验证 72、冻结测试 60。

## 7. 验收规则

1. 基线与候选必须使用相同模型、tokenizer、数据、硬件、批量、预热和统计窗口。
2. 必须先通过精度门禁，再评价性能收益。
3. 正式结果绑定运行 ID、时间戳、原始路径及 SHA-256。
4. 缺失或冲突证据输出 `insufficient_evidence`，不得由模型补值。
5. 历史失败与拒绝候选保留，不能删除或覆盖。

## 8. 当前验证结果

### 8.1 精度对齐

- AscendC 与 Triton 均完成 100 步并正常退出。
- 最大逐步 Loss 相对差：0.0017596037（阈值 0.02）。
- 末窗口 Loss 相对差：0.0002319596（阈值 0.01）。
- Loss 曲线 Pearson：0.9999651705（阈值 0.99）。
- 结论：历史冻结精度证据 `PASS`。

### 8.2 Shengteng 续训和性能复跑

- P2-C04R 成功续训至第 60 步，退出码 0。
- 工程回归：61 passed in 18.35s。
- 三组独立 P50 时延改善：37.83%、38.31%、35.54%。
- 平均改善：37.23%；95% CI：33.54%–40.91%。
- 三组吞吐提升：60.86%、62.11%、55.13%。
- 六个性能子任务退出码全部为 0。

### 8.3 冻结测试

对保存的 60 条冻结测试预测使用当前评分器重新评分：decision accuracy=1.0、issue macro-F1=1.0、unsafe-pass rate=0、valid-output rate=1.0。该结果是保存预测的确定性复评分，不是本轮第 60 步检查点重新推理。

## 9. 证据与复现

提交包包含原始日志 ZIP、Skill ZIP、源代码 ZIP、精度报告、性能报告和 SHA-256 清单。所有正式数字均可追溯至 Shengteng 服务器冻结快照及最终复跑结果文件。
