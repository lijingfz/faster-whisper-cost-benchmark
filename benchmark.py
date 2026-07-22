#!/usr/bin/env python3
"""
faster-whisper ASR 成本基准 · 主入口

在任意 EC2 GPU 实例上运行：
  1. 自动检测实例类型 / region / GPU
  2. 扫描 batch_size，找到使 RTF（吞吐）最大的最优值（离线批量吞吐场景）
  3. 结合 On-Demand 时价，算出每音频小时成本
  4. 输出该实例的 JSON + Markdown 成本分析报告到 results/

示例：
  python benchmark.py                          # 全自动
  python benchmark.py --price 2.52             # 手动指定时价
  python benchmark.py --batch-sizes 1,4,8,16,32,48 --runs 3
  python benchmark.py --audio audio/my.mp3     # 用自定义音频
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import gpu_info, pricing, report  # noqa: E402
from src.bench_core import sweep_batch_sizes, pick_optimal, transcribe_once  # noqa: E402

DEFAULT_MODEL = "large-v3"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BATCH_SIZES = "1,4,8,16,24,32,48"
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
    p = argparse.ArgumentParser(description="faster-whisper ASR 成本基准")
    p.add_argument("--audio", default=DEFAULT_AUDIO, help="测试音频路径")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                   help="固定 float16 以便跨实例公平对比（Blackwell 上 INT8 不可用）")
    p.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    p.add_argument("--runs", type=int, default=3, help="每档计时次数取中位数")
    p.add_argument("--language", default="en", help="固定语种避免每次检测波动；none=自动")
    p.add_argument("--price", type=float, default=None,
                   help="手动指定 On-Demand 时价(美元/小时)，最高优先级")
    p.add_argument("--instance-type", default=None, help="覆盖自动检测的实例类型")
    p.add_argument("--region", default=None, help="覆盖自动检测的 region")
    p.add_argument("--prices-file", default=os.path.join(HERE, "prices.json"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--exhaustive", action="store_true",
                   help="扫描全部 batch_size，不因饱和提前停止")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        sys.exit(f"[错误] 找不到音频 {args.audio}。请先运行 prepare_audio.py 生成，"
                 f"或用 --audio 指定。")

    language = None if args.language.lower() == "none" else args.language
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    instance_type = gpu_info.get_instance_type(args.instance_type)
    region = gpu_info.get_region(args.region)
    gpu = gpu_info.get_gpu_info()
    duration = ffprobe_duration(args.audio)

    print("=" * 60)
    print(f"实例类型 : {instance_type}   Region: {region}")
    print(f"GPU      : {gpu.name}  {gpu.compute_capability}  "
          f"{gpu.total_vram_gib}GiB")
    print(f"模型     : {args.model}  compute_type={args.compute_type}")
    print(f"音频     : {os.path.basename(args.audio)}  ({duration:.1f}s)")
    print(f"batch_size 候选: {batch_sizes}  每档 {args.runs} 次取中位数")
    print("=" * 60)

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    print("加载模型...")
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    pipeline = BatchedInferencePipeline(model=model)

    print("预热运行（不计入结果）...")
    transcribe_once(pipeline, args.audio, batch_size=min(8, max(batch_sizes)),
                    language=language)

    print("\n开始扫描 batch_size：")
    patience = 10 ** 9 if args.exhaustive else 2
    results = sweep_batch_sizes(
        pipeline, args.audio, duration, batch_sizes, args.runs, language,
        early_stop_patience=patience,
    )
    optimal = pick_optimal(results)

    price, source = pricing.get_on_demand_price(
        instance_type, region, args.prices_file, args.price)

    data = report.build_result_dict(
        instance_type=instance_type, region=region, gpu=gpu_info.as_dict(gpu),
        model=args.model, compute_type=args.compute_type, audio_path=args.audio,
        audio_duration_s=duration, runs=args.runs, price_per_hour=price,
        price_source=source, results=results, optimal=optimal,
    )

    slug = slugify(instance_type)
    json_path = os.path.join(args.out_dir, f"{slug}.json")
    md_path = os.path.join(args.out_dir, f"{slug}.md")
    report.write_json(data, json_path)
    report.write_markdown(data, md_path)

    print("\n" + "=" * 60)
    if optimal:
        print(f"最优 batch_size = {optimal.batch_size}   RTF = {optimal.rtf:.1f}x")
    if data["cost_per_audio_hour_usd"] is not None:
        c = data["cost_per_audio_hour_usd"]
        print(f"On-Demand ${price:.4f}/hr（{source}）  =>  "
              f"每音频小时成本 ${c:.4f}  (每1000音频小时 ${c*1000:.2f})")
    else:
        print(f"价格来源: {source}（无法算成本，可用 --price 指定）")
    print(f"报告已写入:\n  {json_path}\n  {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
