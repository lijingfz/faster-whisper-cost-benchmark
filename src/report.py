"""单实例结果的 JSON / Markdown 输出。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from .bench_core import BatchResult, SweepSummary


def _opt_dict(r: Optional[BatchResult]) -> Optional[dict]:
    if not r:
        return None
    return {
        "batch_size": r.batch_size,
        "median_time_s": r.median_time_s,
        "rtf": r.rtf,
        "peak_vram_gib": r.peak_vram_gib,
    }


def build_result_dict(
    instance_type: str,
    region: str,
    gpu: dict,
    model: str,
    compute_type: str,
    audio_path: str,
    audio_duration_s: float,
    runs: int,
    price_per_hour: Optional[float],
    price_source: str,
    results: List[BatchResult],
    optimal: Optional[BatchResult],
    saturating: Optional[BatchResult] = None,
    summary: Optional[SweepSummary] = None,
    total_vram_gib: Optional[float] = None,
    audio_multiplier: Optional[int] = None,
    iterations: int = 1,
    saturate_mode: bool = True,
    vram_target_frac: float = 0.90,
) -> dict:
    def cost_of(r: Optional[BatchResult]) -> Optional[float]:
        if r and r.rtf and price_per_hour is not None:
            # $/audio-hour = 时价 / RTF（RTF = 每墙钟小时可转写的音频小时数）
            return price_per_hour / r.rtf
        return None

    cost_per_audio_hour = cost_of(optimal)

    if summary is not None:
        sat_status = ("oom" if summary.oom
                      else ("saturated" if summary.saturated else "not_saturated"))
    else:
        sat_status = "unknown"

    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance_type": instance_type,
        "region": region,
        "gpu": gpu,
        "model": model,
        "compute_type": compute_type,
        "audio": os.path.basename(audio_path),
        "audio_duration_s": audio_duration_s,
        "audio_multiplier": audio_multiplier,
        "runs_per_size": runs,
        "adaptive": {
            "saturate_mode": saturate_mode,
            "vram_target_frac": vram_target_frac,
            "iterations": iterations,
            "total_vram_gib": total_vram_gib,
            "peak_vram_gib": (summary.peak_vram_gib if summary else None),
            "vram_utilization": (summary.vram_utilization if summary else None),
            "saturation_status": sat_status,
        },
        "price": {
            "on_demand_usd_per_hour": price_per_hour,
            "source": price_source,
        },
        "optimal": _opt_dict(optimal),
        "saturating": _opt_dict(saturating),
        "cost_per_audio_hour_usd": cost_per_audio_hour,
        "cost_per_audio_hour_usd_at_saturation": cost_of(saturating),
        "sweep": [
            {
                "batch_size": r.batch_size,
                "median_time_s": r.median_time_s,
                "rtf": r.rtf,
                "peak_vram_gib": r.peak_vram_gib,
                "status": r.status,
                "times": r.times,
            }
            for r in results
        ],
        "text_head": (optimal.text_head if optimal else
                      (saturating.text_head if saturating else "")),
    }


def write_json(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fmt(v, spec="", dash="—"):
    return format(v, spec) if v is not None else dash


_SAT_LABEL = {
    "oom": "已打爆（OOM，达到显存物理极限）",
    "saturated": "已占满（达到显存目标阈值）",
    "not_saturated": "未占满（受限于音频段数或 batch 上限）",
    "unknown": "未知",
}


def write_markdown(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    g = data["gpu"]
    price = data["price"]["on_demand_usd_per_hour"]
    opt = data["optimal"]
    sat = data.get("saturating")
    cost = data["cost_per_audio_hour_usd"]
    sat_cost = data.get("cost_per_audio_hour_usd_at_saturation")
    ad = data.get("adaptive", {})

    lines = []
    lines.append(f"# 成本分析报告 · {data['instance_type']}")
    lines.append("")
    lines.append(f"> 生成时间(UTC)：{data['timestamp_utc']}")
    lines.append("> 场景：离线批量吞吐 ｜ 策略：自适应把显存推到极限后测 RTF / 成本 / 耗时")
    lines.append("")
    lines.append("## 环境")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 实例类型 | {data['instance_type']} |")
    lines.append(f"| Region | {data['region']} |")
    lines.append(f"| GPU | {g['name']} ({g['compute_capability']}, "
                 f"{g['total_vram_gib']}GiB) |")
    lines.append(f"| 模型 | {data['model']} |")
    lines.append(f"| compute_type | {data['compute_type']} |")
    mult = data.get("audio_multiplier")
    mult_str = f"（{mult}× base）" if mult else ""
    lines.append(f"| 测试音频 | {data['audio']} "
                 f"({data['audio_duration_s']:.1f}s{mult_str}) |")
    lines.append(f"| 每档计时次数 | {data['runs_per_size']}（取中位数）|")
    lines.append("")

    # 显存占满情况
    lines.append("## 显存占满情况")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 模式 | {'显存占满(saturate)' if ad.get('saturate_mode') else 'RTF饱和提前停止'} |")
    lines.append(f"| 占满阈值 | {_fmt(ad.get('vram_target_frac'), '.0%')} |")
    lines.append(f"| 峰值显存 | {_fmt(ad.get('peak_vram_gib'), '.2f')} / "
                 f"{_fmt(ad.get('total_vram_gib'), '.2f')} GiB |")
    lines.append(f"| 显存利用率 | {_fmt(ad.get('vram_utilization'), '.0%')} |")
    lines.append(f"| 饱和状态 | {_SAT_LABEL.get(ad.get('saturation_status'), '未知')} |")
    lines.append(f"| 自适应迭代轮数 | {ad.get('iterations', 1)}（段饥饿时自动增大音频）|")
    lines.append("")

    lines.append("## 核心结论")
    lines.append("")
    if opt:
        lines.append(f"- **成本最优 batch_size**：**{opt['batch_size']}**"
                     f"（RTF 最大 ⇒ 单位成本最低）")
        lines.append(f"  - RTF（倍速）：**{_fmt(opt['rtf'], '.1f')}x**，"
                     f"中位推理耗时 {_fmt(opt['median_time_s'], '.2f')}s，"
                     f"峰值显存 {_fmt(opt['peak_vram_gib'], '.2f')} GiB")
    else:
        lines.append("- 未能得到有效的最优结果（可能全部 OOM 或出错）。")
    if sat:
        lines.append(f"- **显存占满 batch_size**：**{sat['batch_size']}**"
                     f"（峰值显存最大档）")
        lines.append(f"  - RTF（倍速）：{_fmt(sat['rtf'], '.1f')}x，"
                     f"中位推理耗时 {_fmt(sat['median_time_s'], '.2f')}s，"
                     f"峰值显存 {_fmt(sat['peak_vram_gib'], '.2f')} GiB")
        if sat_cost is not None:
            lines.append(f"  - 占满档每音频小时成本：${sat_cost:.4f} / audio-hour")
    lines.append(f"- On-Demand 价格："
                 f"{'$' + _fmt(price, '.4f') + '/hr' if price is not None else '未知'}"
                 f"（来源：{data['price']['source']}）")
    if cost is not None:
        lines.append(f"- **每音频小时成本（成本最优档）**：**${cost:.4f} / audio-hour** "
                     f"= 时价 ÷ RTF")
        lines.append(f"- 每 1000 音频小时成本：约 **${cost * 1000:.2f}**")
    else:
        lines.append("- 每音频小时成本：无法计算（缺价格或无有效 RTF）。"
                     "可用 `--price` 手动指定时价。")
    lines.append("")

    lines.append("## batch_size 扫描明细")
    lines.append("")
    lines.append("| batch_size | 中位耗时(s) | RTF(倍速) | 峰值显存(GiB) | 状态 |")
    lines.append("|---:|---:|---:|---:|---|")
    for r in data["sweep"]:
        marks = ""
        if opt and r["batch_size"] == opt["batch_size"]:
            marks += " ⭐"
        if sat and r["batch_size"] == sat["batch_size"]:
            marks += " 🔥"
        lines.append(
            f"| {r['batch_size']}{marks} | {_fmt(r['median_time_s'], '.2f')} | "
            f"{_fmt(r['rtf'], '.1f')} | {_fmt(r['peak_vram_gib'], '.2f')} | "
            f"{r['status']} |"
        )
    lines.append("")
    lines.append("> ⭐ = 成本最优（RTF 最大）档；🔥 = 显存占满（峰值显存最大）档。")
    lines.append("")
    if data.get("text_head"):
        lines.append("## 转写抽样（一致性检查）")
        lines.append("")
        lines.append("> " + data["text_head"] + " ...")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
