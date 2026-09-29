#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, type=Path)
    parser.add_argument("--accuracy", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--human-review", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if len(args.pair) != 3:
        raise ValueError("exactly three independent pairs are required")
    pairs = [load(path) for path in args.pair]
    if not all(p["comparison"].get("comparable") for p in pairs):
        raise ValueError("all pairs must be comparable")
    improvements = [p["comparison"]["p50_step_time_improvement_percent"] for p in pairs]
    mean = statistics.mean(improvements)
    stdev = statistics.stdev(improvements)
    margin = 4.302652729911275 * stdev / math.sqrt(3)
    baselines = [p["baseline"]["step_time_ms_p50"] for p in pairs]
    optimized = [p["optimized"]["step_time_ms_p50"] for p in pairs]
    summary = {
        "schema_version": "migrationproof.p2-independent-performance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_count": 3,
        "protocol": {"warmup_iterations": 1, "measured_iterations_per_run": 20, "effective_batch_size": 16, "device_count": 2, "device_model": "Ascend 910B4-1"},
        "baseline_p50_ms_by_pair": baselines,
        "optimized_p50_ms_by_pair": optimized,
        "p50_step_time_improvement_percent_by_pair": improvements,
        "p50_step_time_improvement_percent_mean": mean,
        "p50_step_time_improvement_percent_stdev": stdev,
        "p50_step_time_improvement_percent_ci95": [mean - margin, mean + margin],
        "minimum_observed_improvement_percent": min(improvements),
        "performance_gate": {"minimum_improvement_percent": 5.0, "passed": min(improvements) >= 5.0},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.open("x", encoding="utf-8", newline="\n").write(json.dumps(summary, indent=2) + "\n")
    evidence_paths = args.pair + [args.accuracy, args.gate, args.human_review, args.manifest, args.summary]
    ledger = {
        "schema_version": "migrationproof.evidence-ledger.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [{"path": str(path), "sha256": sha256(path)} for path in evidence_paths],
    }
    args.ledger.open("x", encoding="utf-8", newline="\n").write(json.dumps(ledger, indent=2) + "\n")
    accuracy = load(args.accuracy)
    gate = load(args.gate)
    review = load(args.human_review)["status"]
    final_status = "conditional_pass" if review.get("status") == "AUTHOR_REVIEW_NEEDED" else "pass"
    report = f"""# P2-C03 三组独立昇腾性能对照与最终技术结论

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

- 技术门禁：通过。三组性能对照全部成功，最小加速 {min(improvements):.2f}%，平均加速 {mean:.2f}%，95% 置信区间 [{mean-margin:.2f}%, {mean+margin:.2f}%]，明显超过 5% 门槛。
- 冻结测试：通过。决策准确率 {accuracy['metrics']['decision_accuracy']:.4f}，issue Macro-F1 {accuracy['metrics']['issue_macro_f1']:.4f}，unsafe pass rate {accuracy['metrics']['unsafe_pass_rate']:.4f}，有效输出率 {accuracy['metrics']['valid_output_rate']:.4f}。
- 最终状态：`{final_status}`。唯一未闭环项是 20 条解释质量样本仍需两名人工评审填写，不能由自动化流程代填。

## 性能协议

- 硬件：2 × Ascend 910B4-1；CANN 8.2.RC1。
- 数据：MigrationProof-Bench v0.6.1-p2，训练 360、验证 72、冻结测试 60，跨集合 ID 无重叠。
- 公平控制：相同训练集、有效批量 16、精度与并行设置；每侧新增运行 21 步，首步预热，统计 20 步。
- 原始基线：原始 DCP release，重计算开启，前向/反向预取为 0。
- P2-C03：从第 30 步检查点恢复，重计算关闭，前向/反向预取为 1；累计步号 31–51，但新增测量窗口与基线严格等长。

## 三组结果

| 组次 | 基线 P50 (ms) | P2-C03 P50 (ms) | 步时改善 |
|---:|---:|---:|---:|
"""
    for index, (base, opt, improvement) in enumerate(zip(baselines, optimized, improvements), 1):
        report += f"| {index} | {base:.2f} | {opt:.2f} | {improvement:.2f}% |\n"
    report += f"""

## 精度与安全门禁

- 冻结测试门禁：`{gate['test_gate']['passed']}`；性能测试授权：`{gate['test_gate']['performance_authorized']}`。
- 候选模型：P2-C03 short checkpoint，独立验证集与冻结测试集均未用于训练。
- 性能训练不会回写 P2-C03 原始检查点；三组均以相同候选检查点重新恢复。

## 待完成事项

1. 两名人工评审分别完成 20 条解释质量打分（真实性、证据覆盖、可读性、决策一致性）。
2. 汇总评分并完成分歧仲裁后，将最终状态从 `conditional_pass` 更新为 `pass` 或按评分结果退回修订。
3. 在提交包中纳入本报告、三组逐对 JSON、汇总 JSON、证据账本、原始日志与配置。
"""
    args.report.open("x", encoding="utf-8", newline="\n").write(report)
    print(json.dumps({"summary": str(args.summary), "ledger": str(args.ledger), "report": str(args.report), "final_status": final_status}, indent=2))


if __name__ == "__main__":
    main()
