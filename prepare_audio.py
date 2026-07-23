#!/usr/bin/env python3
"""
准备标准测试音频：把一个基准音频重复拼接到目标时长/倍数，供跨实例公平对比。

为什么要拼长：BatchedInferencePipeline 用 VAD 把音频切成语音段后按 batch 并行解码。
large-v3 按 30s 窗口工作，音频太短切出的段数少于 batch_size，大 batch「填不满」，
既测不出真实吞吐，也无法把显存推到物理极限（段饥饿）。跨实例对比只要所有机器用
同一份音频即可（RTF 与绝对时长无关）。

用法：
  python prepare_audio.py                              # 用自带 base 音频拼到 32 倍
  python prepare_audio.py --multiplier 64              # 拼成 base 的 64 倍
  python prepare_audio.py --input my.wav --target 900  # 按目标时长(秒)拼接
  python prepare_audio.py --repeat 10                  # 直接指定重复次数

benchmark.py 的自适应流程会直接调用本文件的 build_audio() 在运行中重新生成更长音频。
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "audio", "base-sample.mp3")
DEFAULT_OUTPUT = os.path.join(HERE, "audio", "benchmark-audio.mp3")
DEFAULT_MULTIPLIER = 20  # 起始倍数：足以喂满大 batch，避免段饥饿


def duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.decode().strip())


def build_audio(input_path: str, output_path: str, repeat: int) -> float:
    """把 input_path 无损拼接 repeat 次到 output_path，返回输出时长(秒)。

    使用 ffmpeg concat + stream copy，不重编码、无音质损失，速度快。
    可被 benchmark.py 的自适应流程直接调用以在运行中重新生成更长音频。
    """
    repeat = max(1, int(repeat))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    list_path = os.path.join(os.path.dirname(output_path) or ".", "_concat_list.txt")
    abs_in = os.path.abspath(input_path)
    with open(list_path, "w") as f:
        for _ in range(repeat):
            f.write(f"file '{abs_in}'\n")
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", output_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return duration(output_path)


def resolve_repeat(base_s: float, multiplier: int | None,
                   target_s: float | None, repeat: int | None) -> int:
    """按优先级 repeat > multiplier > target 解析出重复次数。"""
    if repeat:
        return max(1, int(repeat))
    if multiplier:
        return max(1, int(multiplier))
    if target_s:
        return max(1, math.ceil(target_s / base_s))
    return DEFAULT_MULTIPLIER


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help="基准音频")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--multiplier", type=int, default=None,
                   help=f"拼成 base 音频的多少倍（默认 {DEFAULT_MULTIPLIER}）")
    p.add_argument("--target", type=float, default=None,
                   help="目标时长(秒)，按需向上取整重复次数（低于 multiplier 优先级）")
    p.add_argument("--repeat", type=int, default=None,
                   help="直接指定重复次数（最高优先级）")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[错误] 找不到基准音频 {args.input}。请放置一个音频文件，"
                 f"或用 --input 指定。")

    base = duration(args.input)
    repeat = resolve_repeat(base, args.multiplier, args.target, args.repeat)
    print(f"基准音频 {os.path.basename(args.input)} 时长 {base:.1f}s，"
          f"重复 {repeat} 次 => 约 {base * repeat:.1f}s")

    actual = build_audio(args.input, args.output, repeat)
    print(f"已生成 {args.output}，实际时长 {actual:.1f}s（{repeat}× base）")


if __name__ == "__main__":
    main()
