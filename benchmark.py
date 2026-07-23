#!/usr/bin/env python3
"""
faster-whisper ASR 成本基准 · 主入口（自适应显存占满版）

在任意 EC2 GPU 实例上运行，全自动完成：
  1. 自动检测实例类型 / region / GPU
  2. 从 base 音频的 32 倍起步，扫描 batch_size 并一路上探，把显存推到 ~90% 或打爆(OOM)
  3. 若显存因语音段不足无法占满（段饥饿），自动增大音频再试，直至占满或触及上限
  4. 在「占满」状态下计算 RTF（吞吐）、推理耗时与每音频小时成本
  5. 输出该实例的 JSON + Markdown 成本分析报告到 results/

示例：
  python benchmark.py                          # 全自动（32× 起步，占满显存）
  python benchmark.py --price 2.52             # 手动指定时价
  python benchmark.py --audio-multiplier 64    # 从 64× 起步
  python benchmark.py --no-saturate            # 退回旧的「RTF 饱和提前停止」模式
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import gpu_info, pricing, report  # noqa: E402
from src.bench_core import (  # noqa: E402
    sweep_batch_sizes, summarize_sweep, pick_optimal, pick_saturating,
    transcribe_once,
)
from prepare_audio import build_audio, DEFAULT_INPUT, DEFAULT_MULTIPLIER  # noqa: E402

DEFAULT_MODEL = "large-v3"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BATCH_SIZES = "1,4,8,16,32,48,64,80,96"
DEFAULT_AUDIO = os.path.join(HERE, "audio", "benchmark-audio.mp3")


def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.decode().strip())


def slugify(instance_type: str) -> str:
    return instance_type.replace(".", "-")


def main():
    p = argparse.ArgumentParser(description="faster-whisper ASR 成本基准（自适应显存占满）")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                   help="固定 float16 以便跨实例公平对比（Blackwell 上 INT8 不可用）")
    p.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES,
                   help="起始候选 batch；saturate 模式会在此基础上自动向上翻倍探顶")
    p.add_argument("--runs", type=int, default=1, help="每档计时次数取中位数")
    p.add_argument("--language", default="en", help="固定语种避免每次检测波动；none=自动")
    p.add_argument("--price", type=float, default=None,
                   help="手动指定 On-Demand 时价(美元/小时)，最高优先级")
    p.add_argument("--instance-type", default=None, help="覆盖自动检测的实例类型")
    p.add_argument("--region", default=None, help="覆盖自动检测的 region")
    p.add_argument("--prices-file", default=os.path.join(HERE, "prices.json"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))

    # 音频与自适应占满相关
    p.add_argument("--base-audio", default=DEFAULT_INPUT,
                   help="用于拼接的基准音频（自适应流程按倍数重新生成）")
    p.add_argument("--audio", default=None,
                   help="直接指定测试音频，跳过自动生成与自适应增大")
    p.add_argument("--audio-multiplier", type=int, default=DEFAULT_MULTIPLIER,
                   help=f"起始音频 = base 的多少倍（默认 {DEFAULT_MULTIPLIER}）")
    p.add_argument("--vram-target-frac", type=float, default=0.90,
                   help="显存利用率达到该比例即视为『占满』（默认 0.90）")
    p.add_argument("--audio-grow-factor", type=float, default=2.0,
                   help="段饥饿时音频倍数的增长系数（默认 2.0）")
    p.add_argument("--max-audio-multiplier", type=int, default=1024,
                   help="音频倍数安全上限，防止无限增大（默认 1024）")
    p.add_argument("--max-batch", type=int, default=1024,
                   help="batch 上探安全上限（默认 1024）")
    p.add_argument("--max-iterations", type=int, default=6,
                   help="自适应增大音频的最大轮数（默认 6）")
    p.add_argument("--no-saturate", dest="saturate", action="store_false",
                   help="退回旧模式：RTF 饱和提前停止，不追求占满显存")
    p.set_defaults(saturate=True)
    args = p.parse_args()

    language = None if args.language.lower() == "none" else args.language
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    instance_type = gpu_info.get_instance_type(args.instance_type)
    region = gpu_info.get_region(args.region)
    gpu = gpu_info.get_gpu_info()
    total_vram = gpu.total_vram_gib

    print("=" * 64)
    print(f"实例类型 : {instance_type}   Region: {region}")
    print(f"GPU      : {gpu.name}  {gpu.compute_capability}  {total_vram}GiB")
    print(f"模型     : {args.model}  compute_type={args.compute_type}")
    print(f"模式     : {'显存占满(saturate)' if args.saturate else 'RTF饱和提前停止'}"
          f"   显存占满阈值 {args.vram_target_frac:.0%}")
    print(f"起始候选 batch: {batch_sizes}  每档 {args.runs} 次取中位数")
    print("=" * 64)

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    print("加载模型...")
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    pipeline = BatchedInferencePipeline(model=model)

    # ---- 准备音频：用户显式指定则直接用，否则按倍数生成并允许自适应增大 ----
    fixed_audio = args.audio is not None
    if fixed_audio:
        audio_path = args.audio
        if not os.path.exists(audio_path):
            sys.exit(f"[错误] 找不到音频 {audio_path}。")
        multiplier = None
    else:
        if not os.path.exists(args.base_audio):
            sys.exit(f"[错误] 找不到基准音频 {args.base_audio}，无法生成测试音频。")
        audio_path = DEFAULT_AUDIO
        multiplier = args.audio_multiplier

    # 预热（用较小 batch）
    def warmup(path):
        print("预热运行（不计入结果）...")
        transcribe_once(pipeline, path, batch_size=min(8, max(batch_sizes)),
                        language=language)

    # ---- 自适应闭环：扫描 -> 未占满且段饥饿则增大音频 -> 重扫 ----
    iteration = 0
    results = []
    summary = None
    duration = 0.0
    while True:
        iteration += 1
        if not fixed_audio:
            print(f"\n[迭代 {iteration}] 生成音频：base × {multiplier} ...")
            duration = build_audio(args.base_audio, audio_path, multiplier)
        else:
            duration = ffprobe_duration(audio_path)
        print(f"[迭代 {iteration}] 测试音频时长 {duration:.1f}s"
              f"{'' if fixed_audio else f'（{multiplier}× base）'}")

        warmup(audio_path)
        print(f"[迭代 {iteration}] 开始扫描 batch_size：")
        patience = 2  # 仅在非 saturate 模式生效
        results = sweep_batch_sizes(
            pipeline, audio_path, duration, batch_sizes, args.runs, language,
            early_stop_patience=patience, saturate=args.saturate,
            max_batch=args.max_batch,
        )
        summary = summarize_sweep(results, total_vram, args.vram_target_frac)
        print(f"[迭代 {iteration}] 峰值显存 {summary.peak_vram_gib:.2f}/{total_vram}GiB "
              f"（利用率 {summary.vram_utilization:.0%}）  "
              f"{'已OOM打爆' if summary.oom else ('已占满' if summary.saturated else '未占满')}")

        # 结束条件：非 saturate、固定音频、已饱和、达轮数上限、达倍数上限、或非段饥饿
        if not args.saturate or fixed_audio or summary.saturated:
            break
        if iteration >= args.max_iterations:
            print(f"[停止] 达到最大迭代轮数 {args.max_iterations}，仍未占满显存。")
            break
        if not summary.starved:
            # 显存仍随 batch 上涨但未达阈值，且 batch 已探至上限：增大音频通常无益
            print("[停止] 显存未达阈值但也非段饥饿（batch 已探顶），停止。")
            break
        new_mult = int(multiplier * args.audio_grow_factor)
        if new_mult <= multiplier:
            new_mult = multiplier + 1
        if new_mult > args.max_audio_multiplier:
            print(f"[停止] 音频倍数将超过上限 {args.max_audio_multiplier}，停止增大。")
            break
        print(f"[自适应] 检测到语音段不足（显存平台且未占满），"
              f"音频倍数 {multiplier} → {new_mult}，重新扫描。")
        multiplier = new_mult

    optimal = pick_optimal(results)
    saturating = pick_saturating(results)

    price, source = pricing.get_on_demand_price(
        instance_type, region, args.prices_file, args.price)

    data = report.build_result_dict(
        instance_type=instance_type, region=region, gpu=gpu_info.as_dict(gpu),
        model=args.model, compute_type=args.compute_type, audio_path=audio_path,
        audio_duration_s=duration, runs=args.runs, price_per_hour=price,
        price_source=source, results=results, optimal=optimal,
        saturating=saturating, summary=summary, total_vram_gib=total_vram,
        audio_multiplier=multiplier, iterations=iteration, saturate_mode=args.saturate,
        vram_target_frac=args.vram_target_frac,
    )

    slug = slugify(instance_type)
    json_path = os.path.join(args.out_dir, f"{slug}.json")
    md_path = os.path.join(args.out_dir, f"{slug}.md")
    report.write_json(data, json_path)
    report.write_markdown(data, md_path)

    print("\n" + "=" * 64)
    print(f"显存占满：峰值 {summary.peak_vram_gib:.2f}/{total_vram}GiB "
          f"({summary.vram_utilization:.0%})  "
          f"{'OOM打爆' if summary.oom else ('已占满' if summary.saturated else '未占满')}")
    if optimal:
        print(f"成本最优 batch_size = {optimal.batch_size}   "
              f"RTF = {optimal.rtf:.1f}x   中位耗时 {optimal.median_time_s:.2f}s")
    if saturating:
        print(f"显存占满 batch_size = {saturating.batch_size}   "
              f"RTF = {saturating.rtf:.1f}x   中位耗时 {saturating.median_time_s:.2f}s   "
              f"峰值显存 {saturating.peak_vram_gib:.2f}GiB")
    if data["cost_per_audio_hour_usd"] is not None:
        c = data["cost_per_audio_hour_usd"]
        print(f"On-Demand ${price:.4f}/hr（{source}）  =>  "
              f"每音频小时成本 ${c:.4f}  (每1000音频小时 ${c*1000:.2f})")
    else:
        print(f"价格来源: {source}（无法算成本，可用 --price 指定）")
    print(f"报告已写入:\n  {json_path}\n  {md_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
