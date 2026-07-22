"""单实例结果的 JSON / Markdown 输出。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from .bench_core import BatchResult


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
) -> dict:
    cost_per_audio_hour = None
    if optimal and optimal.rtf and price_per_hour is not None:
        # $/audio-hour = 时价 / RTF（RTF = 每墙钟小时可转写的音频小时数）
        cost_per_audio_hour = price_per_hour / optimal.rtf
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance_type": instance_type,
        "region": region,
        "gpu": gpu,
        "model": model,
        "compute_type": compute_type,
        "audio": os.path.basename(audio_path),
        "audio_duration_s": audio_duration_s,
        "runs_per_size": runs,
        "price": {
            "on_demand_usd_per_hour": price_per_hour,
            "source": price_source,
        },
        "optimal": None if not optimal else {
            "batch_size": optimal.batch_size,
            "median_time_s": optimal.median_time_s,
            "rtf": optimal.rtf,
            "peak_vram_gib": optimal.peak_vram_gib,
        },
        "cost_per_audio_hour_usd": cost_per_audio_hour,
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
        "text_head": (optimal.text_head if optimal else ""),
    }


def write_json(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fmt(v, spec="", dash="—"):
    return format(v, spec) if v is not None else dash


def write_markdown(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    g = data["gpu"]
    price = data["price"]["on_demand_usd_per_hour"]
    opt = data["optimal"]
    cost = data["cost_per_audio_hour_usd"]

    lines = []
    lines.append(f"# 成本分析报告 · {data['instance_type']}")
    lines.append("")
    lines.append(f"> 生成时间(UTC)：{data['timestamp_utc']}")
    lines.append(f"> 场景：离线批量吞吐（最优 batch_size 追求最大 RTF / 最低单位成本）")
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
    lines.append(f"| 测试音频 | {data['audio']} ({data['audio_duration_s']:.1f}s) |")
    lines.append(f"| 每档计时次数 | {data['runs_per_size']}（取中位数）|")
    lines.append("")

    lines.append("## 核心结论")
    lines.append("")
    if opt:
        lines.append(f"- **最优 batch_size**：**{opt['batch_size']}**")
        lines.append(f"- **RTF（倍速）**：**{_fmt(opt['rtf'], '.1f')}x**")
        lines.append(f"- 该档峰值显存：{_fmt(opt['peak_vram_gib'], '.2f')} GiB")
    else:
        lines.append("- 未能得到有效的最优结果（可能全部 OOM 或出错）。")
    lines.append(f"- On-Demand 价格："
                 f"{'$' + _fmt(price, '.4f') + '/hr' if price is not None else '未知'}"
                 f"（来源：{data['price']['source']}）")
    if cost is not None:
        lines.append(f"- **每音频小时成本**：**${cost:.4f} / audio-hour** "
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
        mark = " ⭐" if opt and r["batch_size"] == opt["batch_size"] else ""
        lines.append(
            f"| {r['batch_size']}{mark} | {_fmt(r['median_time_s'], '.2f')} | "
            f"{_fmt(r['rtf'], '.1f')} | {_fmt(r['peak_vram_gib'], '.2f')} | "
            f"{r['status']} |"
        )
    lines.append("")
    lines.append("> ⭐ = 选定的最优 batch_size。")
    lines.append("")
    if data.get("text_head"):
        lines.append("## 转写抽样（一致性检查）")
        lines.append("")
        lines.append("> " + data["text_head"] + " ...")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
