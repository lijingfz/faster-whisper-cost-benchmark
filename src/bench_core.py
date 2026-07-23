"""基准核心：显存采样、单次转写计时、batch_size 扫描。

支持两种寻优取向（同一份数据可同时得出）：
- 成本最优：$/audio-hour = 时价 / RTF，因此 RTF 最大的档位单位成本最低。
- 显存饱和（saturate 模式）：把 batch_size 一路上探到 OOM 或显存平台，
  确保测出的是「GPU 真正吃满」时的吞吐，而非因语音段不足造成的假平台。
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


@dataclass
class SweepSummary:
    """一轮扫描（固定音频）的汇总，用于自适应闭环决策。"""
    oom: bool = False                 # 是否触发过 OOM（显存被打爆）
    peak_vram_gib: float = 0.0        # 本轮所有档位的最大峰值显存
    vram_utilization: float = 0.0     # peak_vram / 总显存
    saturated: bool = False           # OOM 或 峰值显存 >= 目标阈值
    vram_climbing: bool = False       # 最大 batch 仍在增大显存（可继续上探 batch）
    starved: bool = False             # 未饱和且显存已平台 => 语音段不足，需增大音频
    max_ok_batch: Optional[int] = None


def sweep_batch_sizes(
    pipeline,
    audio_path: str,
    audio_duration_s: float,
    batch_sizes: List[int],
    runs: int,
    language: Optional[str],
    early_stop_epsilon: float = 0.02,
    early_stop_patience: int = 2,
    saturate: bool = False,
    max_batch: int = 1024,
    vram_plateau_epsilon: float = 0.03,
) -> List[BatchResult]:
    """依次测试各 batch_size。

    普通模式：
      - 每档计时 runs 次取中位数，算 RTF 与峰值显存。
      - 遇 OOM：标记该档 oom 并停止继续增大（更大的一定也 OOM）。
      - 饱和提前停止：连续 early_stop_patience 档 RTF 相对提升 < epsilon 即停。

    saturate 模式（saturate=True）：
      - 关闭 RTF 饱和提前停止；目标是把显存推到极限。
      - 动态上探：扫完给定候选后，若最大档显存相对上一档仍在上涨
        （> vram_plateau_epsilon），自动追加 batch=上一档×2，直到 OOM、
        显存不再上涨（平台）、或触及 max_batch。
      - 遇 OOM：标记并停止（已把显存打爆，即达到硬件极限）。
    """
    results: List[BatchResult] = []
    best_rtf = 0.0
    stagnant = 0
    prev_vram: Optional[float] = None
    queue = sorted(set(int(b) for b in batch_sizes))
    seen = set()

    while queue:
        bs = queue.pop(0)
        if bs in seen or bs > max_batch:
            continue
        seen.add(bs)

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
            print(f"  batch_size={bs:>4}: 中位耗时 {res.median_time_s:8.2f}s  "
                  f"RTF {res.rtf:7.1f}x  峰值显存 {res.peak_vram_gib:6.2f}GiB")
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            if _is_oom(e):
                res.status = "oom"
                print(f"  batch_size={bs:>4}: OOM，显存打爆，停止继续增大 batch")
                results.append(res)
                break
            res.status = "error"
            print(f"  batch_size={bs:>4}: 错误 {e}")
            results.append(res)
            continue
        results.append(res)

        if saturate:
            # 动态上探：仅当已到当前队列末尾时，决定是否追加更大 batch
            if not queue:
                climbing = (prev_vram is None
                            or res.peak_vram_gib > prev_vram * (1 + vram_plateau_epsilon))
                nxt = bs * 2
                if climbing and nxt <= max_batch:
                    queue.append(nxt)
                else:
                    # 显存不再随 batch 上涨（平台）或触顶 => 停止上探
                    if not climbing:
                        print(f"  显存在 batch={bs} 达到平台（增大 batch 不再涨显存），"
                              f"停止上探")
                    break
            prev_vram = res.peak_vram_gib
            continue

        # 普通模式：RTF 饱和提前停止
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


def summarize_sweep(
    results: List[BatchResult],
    total_vram_gib: float,
    vram_target_frac: float = 0.90,
    vram_plateau_epsilon: float = 0.03,
) -> SweepSummary:
    """从一轮扫描结果汇总显存饱和状态，供自适应闭环判断是否需要增大音频。"""
    s = SweepSummary()
    s.oom = any(r.status == "oom" for r in results)
    vrams = [r.peak_vram_gib for r in results if r.peak_vram_gib]
    s.peak_vram_gib = max(vrams) if vrams else 0.0
    s.vram_utilization = (s.peak_vram_gib / total_vram_gib) if total_vram_gib else 0.0

    ok = [r for r in results if r.status == "ok" and r.peak_vram_gib]
    if ok:
        s.max_ok_batch = max(r.batch_size for r in ok)

    s.saturated = s.oom or (s.vram_utilization >= vram_target_frac)

    # 判断最大 batch 处显存是否仍在上涨（用最后两个 ok 档比较）
    if len(ok) >= 2:
        ok_sorted = sorted(ok, key=lambda r: r.batch_size)
        last, prev = ok_sorted[-1].peak_vram_gib, ok_sorted[-2].peak_vram_gib
        s.vram_climbing = last > prev * (1 + vram_plateau_epsilon)
    else:
        s.vram_climbing = True

    # 未饱和 且 显存已平台（不再随 batch 上涨）=> 语音段不足（段饥饿），需增大音频
    s.starved = (not s.saturated) and (not s.vram_climbing)
    return s


def pick_optimal(results: List[BatchResult]) -> Optional[BatchResult]:
    """成本最优 = RTF 最大的成功档位（$/audio-hour = 时价 / RTF）。"""
    ok = [r for r in results if r.status == "ok" and r.rtf]
    if not ok:
        return None
    return max(ok, key=lambda r: r.rtf)


def pick_saturating(results: List[BatchResult]) -> Optional[BatchResult]:
    """显存占满档 = 峰值显存最大的成功档位（用于报告占满时的 RTF/耗时/成本）。"""
    ok = [r for r in results if r.status == "ok" and r.peak_vram_gib]
    if not ok:
        return None
    return max(ok, key=lambda r: r.peak_vram_gib)
