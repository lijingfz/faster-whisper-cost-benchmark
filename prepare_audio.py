#!/usr/bin/env python3
"""
准备标准测试音频：把一个基准音频重复拼接到目标时长，供跨实例公平对比。

为什么要拼长：BatchedInferencePipeline 用 VAD 切段后按 batch 并行，音频太短切出的
段少，大 batch_size 的效果体现不出来。跨实例对比只要所有机器用同一份音频即可
（RTF 与绝对时长无关）。

用法：
  python prepare_audio.py                              # 用自带 base 音频拼到 ~600s
  python prepare_audio.py --input my.wav --target 900  # 自定义输入与目标时长
  python prepare_audio.py --repeat 10                  # 直接指定重复次数
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


def duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return float(out.decode().strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help="基准音频")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--target", type=float, default=600.0,
                   help="目标时长(秒)，按需向上取整重复次数")
    p.add_argument("--repeat", type=int, default=None,
                   help="直接指定重复次数（覆盖 --target）")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[错误] 找不到基准音频 {args.input}。请放置一个音频文件，"
                 f"或用 --input 指定。")

    base = duration(args.input)
    repeat = args.repeat if args.repeat else max(1, math.ceil(args.target / base))
    print(f"基准音频 {os.path.basename(args.input)} 时长 {base:.1f}s，"
          f"重复 {repeat} 次 => 约 {base * repeat:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    list_path = os.path.join(os.path.dirname(args.output) or ".", "_concat_list.txt")
    with open(list_path, "w") as f:
        for _ in range(repeat):
            f.write(f"file '{os.path.abspath(args.input)}'\n")

    # stream copy，不重编码，无音质损失
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", args.output,
    ])
    os.remove(list_path)
    print(f"已生成 {args.output}，实际时长 {duration(args.output):.1f}s")


if __name__ == "__main__":
    main()
