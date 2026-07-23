#!/usr/bin/env python3
"""
汇总多台实例的单实例结果 JSON，生成跨实例成本对比报告。

用法：
  # 把各实例产出的 results/<instance>.json 收集到同一个目录，然后：
  python aggregate_report.py --results-dir results --out results/COST-COMPARISON.md

输出一张按"每音频小时成本"升序排列的对比表，并给出性价比指数
（以最便宜者为 1.00）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load_results(results_dir: str):
    data = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            if "instance_type" in d:
                data.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] 无法解析 {path}: {e}")
    return data


def fmt(v, spec="", dash="—"):
    return format(v, spec) if v is not None else dash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default=os.path.join("results", "COST-COMPARISON.md"))
    args = p.parse_args()

    data = load_results(args.results_dir)
    if not data:
        raise SystemExit(f"在 {args.results_dir} 未找到任何单实例结果 JSON。")

    # 有成本的排前面并按成本升序；无成本的排后面
    def sort_key(d):
        c = d.get("cost_per_audio_hour_usd")
        return (0, c) if c is not None else (1, float("inf"))
    data.sort(key=sort_key)

    costs = [d["cost_per_audio_hour_usd"] for d in data
             if d.get("cost_per_audio_hour_usd") is not None]
    cheapest = min(costs) if costs else None

    lines = []
    lines.append("# 跨实例 ASR 成本对比报告")
    lines.append("")
    lines.append("> 场景：离线批量吞吐 ｜ 指标：每音频小时成本 = On-Demand 时价 ÷ 最优 RTF")
    lines.append("> 性价比指数 = 该实例成本 ÷ 最低成本（1.00 = 最划算）")
    lines.append("")
    lines.append("| 实例 | GPU | 显存 | 显存利用率 | 最优bs | RTF(倍速) | On-Demand $/hr | "
                 "**$/audio-hr** | 每1000音频小时$ | 性价比指数 | 价格来源 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for d in data:
        g = d.get("gpu", {})
        opt = d.get("optimal") or {}
        ad = d.get("adaptive", {})
        price = d.get("price", {}).get("on_demand_usd_per_hour")
        cost = d.get("cost_per_audio_hour_usd")
        idx = (cost / cheapest) if (cost is not None and cheapest) else None
        lines.append(
            f"| {d['instance_type']} | {g.get('name', '?')} | "
            f"{fmt(g.get('total_vram_gib'), '.0f')}GiB | "
            f"{fmt(ad.get('vram_utilization'), '.0%')} | "
            f"{opt.get('batch_size', '—')} | {fmt(opt.get('rtf'), '.1f')} | "
            f"{fmt(price, '.4f')} | **{fmt(cost, '.4f')}** | "
            f"{fmt(cost * 1000 if cost is not None else None, '.2f')} | "
            f"{fmt(idx, '.2f')} | {d.get('price', {}).get('source', '?')} |"
        )
    lines.append("")

    if costs:
        best = data[0]
        lines.append("## 结论")
        lines.append("")
        lines.append(
            f"- **最具性价比**：**{best['instance_type']}**"
            f"（{best.get('gpu', {}).get('name', '?')}），"
            f"每音频小时 ${best['cost_per_audio_hour_usd']:.4f}。")
        lines.append("- 「时价更贵但更快」是否划算，取决于 RTF 提升是否超过价格涨幅；"
                     "本表用 $/audio-hour 直接给出答案。")
        lines.append("")

    lines.append("## 各实例明细")
    lines.append("")
    for d in data:
        slug = d["instance_type"].replace(".", "-")
        lines.append(f"- `{slug}.md` / `{slug}.json`：{d['instance_type']} 详细扫描结果")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"已生成对比报告：{args.out}（共 {len(data)} 台实例）")


if __name__ == "__main__":
    main()
