"""基准核心：显存采样、单次转写计时、batch_size 扫描找最优。

面向"离线批量吞吐"场景：最优 batch_size = 在不 OOM 的前提下使 RTF（吞吐）最大的那个。
因为 每音频小时成本 = 时价 / RTF，RTF 越高单位成本越低。
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


class GpuMemSampler:
    """后台线程按固定间隔采样设备已用显存，记录峰值（GiB）。

    用 torch.cuda.mem_get_info() 做设备级查询，可捕获 CTranslate2 在 torch
    缓存分配器之外分配的显存。"""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_used_bytes = 0

    def _run(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            if used > self.peak_used_bytes:
                self.peak_used_bytes = used
            time.sleep(self.interval_s)

    def __enter__(self):
        self.peak_used_bytes = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()

    @property
    def peak_used_gib(self) -> float:
        return self.peak_used_bytes / (1024 ** 3)


def _is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error" in msg or "cublas" in msg


def transcribe_once(pipeline, audio_path: str, batch_size: int, language: Optional[str]):
    """完整消费 segments 生成器（否则不会真正计算），返回 (耗时秒, 文本开头)。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    segments, _info = pipeline.transcribe(
        audio_path, batch_size=batch_size, language=language
    )
    text_parts = [seg.text for seg in segments]  # 迭代触发实际转写
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, "".join(text_parts).strip()


@dataclass
class BatchResult:
    batch_size: int
    times: List[float] = field(default_factory=list)
    median_time_s: Optional[float] = None
    rtf: Optional[float] = None
    peak_vram_gib: Optional[float] = None
    text_head: str = ""
    status: str = "ok"  # ok | oom | error


def sweep_batch_sizes(
    pipeline,
    audio_path: str,
    audio_duration_s: float,
    batch_sizes: List[int],
    runs: int,
    language: Optional[str],
    early_stop_epsilon: float = 0.02,
    early_stop_patience: int = 2,
) -> List[BatchResult]:
    """依次测试各 batch_size。

    - 每档计时 runs 次取中位数，算 RTF 与峰值显存。
    - 遇 OOM：标记该档 oom 并停止继续增大（更大的一定也 OOM）。
    - 提前停止：若连续 early_stop_patience 档 RTF 相对提升 < epsilon，视为已饱和，
      停止继续扫更大 batch（节省时间）。可用 patience 很大来强制扫全部。
    """
    results: List[BatchResult] = []
    best_rtf = 0.0
    stagnant = 0
    for bs in sorted(set(batch_sizes)):
        res = BatchResult(batch_size=bs)
        try:
            for _ in range(runs):
                with GpuMemSampler() as sampler:
                    elapsed, text = transcribe_once(pipeline, audio_path, bs, language)
                res.times.append(elapsed)
                res.peak_vram_gib = max(res.peak_vram_gib or 0.0, sampler.peak_used_gib)
                if not res.text_head:
                    res.text_head = text[:200]
            res.median_time_s = statistics.median(res.times)
            res.rtf = audio_duration_s / res.median_time_s
            res.status = "ok"
            print(f"  batch_size={bs:>3}: 中位耗时 {res.median_time_s:7.2f}s  "
                  f"RTF {res.rtf:6.1f}x  峰值显存 {res.peak_vram_gib:5.2f}GiB")
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            if _is_oom(e):
                res.status = "oom"
                print(f"  batch_size={bs:>3}: OOM，停止继续增大 batch")
                results.append(res)
                break
            res.status = "error"
            print(f"  batch_size={bs:>3}: 错误 {e}")
            results.append(res)
            continue
        results.append(res)

        # 提前停止判断
        if res.rtf and res.rtf > best_rtf * (1 + early_stop_epsilon):
            best_rtf = max(best_rtf, res.rtf)
            stagnant = 0
        else:
            best_rtf = max(best_rtf, res.rtf or 0.0)
            stagnant += 1
            if stagnant >= early_stop_patience:
                print(f"  RTF 连续 {stagnant} 档提升 < {early_stop_epsilon:.0%}，"
                      f"判定饱和，停止扫描更大 batch")
                break
    return results


def pick_optimal(results: List[BatchResult]) -> Optional[BatchResult]:
    """最优 = RTF 最大的成功档位（离线吞吐目标）。"""
    ok = [r for r in results if r.status == "ok" and r.rtf]
    if not ok:
        return None
    return max(ok, key=lambda r: r.rtf)
