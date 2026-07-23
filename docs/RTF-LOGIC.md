# RTF 计算与自适应显存占满寻优逻辑详解

本文详细说明本项目中 RTF（Real-Time Factor，实时率）的计算方式，以及**自适应把显存
推到极限**（占满 ~90% 或 OOM 打爆）后测 RTF / 成本 / 耗时的完整逻辑。相关实现集中在
`src/bench_core.py`、`benchmark.py` 与 `prepare_audio.py`。

---

## 一、RTF 的定义

本项目采用**吞吐方向**的 RTF 定义（越大越好）：

```
RTF = 音频时长（秒） ÷ 转写墙钟耗时（秒）
```

实现见 `src/bench_core.py` 中：

```python
res.rtf = audio_duration_s / res.median_time_s
```

含义是「每 1 个墙钟小时能转写多少音频小时」。例如 600 秒的音频 12 秒转完，
RTF = 50x。

> 注意：部分文献将 RTF 定义为 处理时间 ÷ 音频时长（越小越好），与本项目的定义
> 互为倒数。本项目报告中统一写作「RTF（倍速）」以避免歧义。

## 二、分子：音频时长

`benchmark.py` 中的 `ffprobe_duration()` 调用 ffprobe 读取音频容器的 duration，
得到 `audio_duration_s`。在一轮扫描内它是固定常数；但在自适应流程里，若发生「段饥饿」
（见第五节），音频会被重新拼长，`audio_duration_s` 随之更新。

**为什么音频要足够长**（见 `prepare_audio.py`）：
`BatchedInferencePipeline` 先用 VAD 把长音频切成语音段，再按 batch_size 并行送入
GPU 解码。`large-v3` 按 30s 窗口工作，**音频太短则切出的段数少于 batch_size**，大
batch「填不满」——不仅测不出真实吞吐，显存也会在远未到硬件极限时就走平（**段饥饿**，
segment starvation）。RTF 是比值，与音频绝对长度无关，只要同一次对比里各实例用同一
份（或等价生成规则的）音频即可公平比较。

> 实测例：275.5s 的 base 音频只拼到 3×（826s）时，G7 在 batch_size≥32 后显存卡在
> 13/31GiB、RTF 卡在 ~132x 不动——这是段饥饿造成的**假平台**，会低估强卡。改用
> 32× 起步、并在需要时自适应加长，才能把显存推到真实极限。（当前默认 20× 起步）

## 三、分母：墙钟耗时的测量

单次计时在 `transcribe_once()`（`src/bench_core.py`），有三个关键细节：

```python
torch.cuda.synchronize()          # ① 起点前同步，清掉未完成的 GPU 队列
t0 = time.perf_counter()
segments, _info = pipeline.transcribe(audio_path, batch_size=batch_size, language=language)
text_parts = [seg.text for seg in segments]   # ② 迭代生成器，触发实际计算
torch.cuda.synchronize()          # ③ 终点前再同步，确保 GPU 真正算完
elapsed = time.perf_counter() - t0
```

1. **两次 `cuda.synchronize()`**：CUDA 调用是异步的。不同步的话，计时器量到的
   只是「发起任务」的时间，而非「计算完成」的时间。
2. **完整消费 segments 生成器**：faster-whisper 的 `transcribe()` 是惰性的，返回
   时几乎没做计算；不迭代 segments，耗时会假到接近 0。这是该库基准测试最常见
   的坑。
3. **计时窗口覆盖全流程**：音频解码、VAD 切段、特征提取、GPU 解码全部计入。
   因此这是端到端吞吐，不是纯 GPU kernel 时间。

## 四、降噪处理

单次计时有波动，项目做了三层控制：

| 手段 | 位置 | 作用 |
|---|---|---|
| 预热运行（不计入结果） | `benchmark.py` | 排除 CUDA context 初始化、kernel 编译、显存分配等一次性开销 |
| 每档跑 `--runs` 次取**中位数**（默认 1 次） | `sweep_batch_sizes()` | 中位数对偶发抖动比平均数更稳健 |
| 固定语种 `--language en` | `benchmark.py` | 跳过每次运行的语种自动检测，消除该部分耗时波动 |

测量期间另有 `GpuMemSampler` 后台线程以 50ms 间隔用 `torch.cuda.mem_get_info()`
做**设备级**显存采样，记录峰值——可捕获 CTranslate2 在 torch 缓存分配器之外分配的
显存。它运行在独立 Python 线程，对 GPU 计时的干扰可忽略。

---

## 五、自适应显存占满：把 GPU 推到极限

本项目的目标不是「浅尝辄止找 RTF 拐点」，而是**把显存推到物理极限（占满 ~90% 或
OOM 打爆），并在该状态下测 RTF / 耗时 / 成本**。为此有两个维度的自动上探：

1. **batch 维度**（`sweep_batch_sizes(..., saturate=True)`）：一路增大 batch 直到显存打满/打爆。
2. **音频维度**（`benchmark.py` 的自适应闭环）：当 batch 已无法再推高显存却仍未占满
   （段饥饿）时，自动加长音频再来一轮。

### 5.1 batch 一维上探（saturate 模式）

显存与 RTF 随 batch_size 变化的物理机制：pipeline 每次取 batch_size 个 VAD 语音段
并行送入 GPU 解码。batch 越大，并行单元填得越满、显存占用越高、RTF 上升；直到显存
接近上限（OOM 边界）或语音段被取空。

`sweep_batch_sizes` 在 `saturate=True` 时：

- **关闭 RTF 饱和提前停止**（不因 RTF 走平就收手，目标是推显存）。
- 先按 `--batch-sizes` 给定的候选（默认 `1,4,8,16,32,48,64,80,96`）逐档测。
- 扫到候选末尾后**动态追加更大 batch**：若最大档显存相对上一档仍上涨
  （> `vram_plateau_epsilon`，默认 3%），追加 `batch×2` 继续；直到
  - 抛出 **CUDA OOM** → 标记该档 `oom`，停止（显存已打爆，达到硬件极限）；或
  - 显存**不再随 batch 上涨**（平台）→ 停止上探（再大也没用）；或
  - 触及 `--max-batch`（默认 1024）安全上限。

### 5.2 显存饱和汇总（`summarize_sweep()`）

一轮扫描结束后，用 `SweepSummary` 汇总：

| 字段 | 含义 |
|---|---|
| `oom` | 是否触发过 OOM（显存被打爆） |
| `peak_vram_gib` | 本轮所有档位的最大峰值显存 |
| `vram_utilization` | `peak_vram / 总显存` |
| `saturated` | `oom` 或 `vram_utilization ≥ vram_target_frac`（默认 0.90） |
| `vram_climbing` | 最后两个 ok 档显存仍在上涨（可继续靠 batch 推高） |
| `starved` | **未饱和 且 显存已平台** ⇒ 语音段不足，需要加长音频 |

### 5.3 音频维度自适应（`benchmark.py` 闭环）

```
multiplier = 20                       # 起始音频 = base × 20（默认）
loop (最多 --max-iterations 轮):
    生成 base × multiplier 的音频
    results = sweep_batch_sizes(..., saturate=True)
    summary = summarize_sweep(results, 总显存, vram_target_frac=0.90)
    if summary.saturated:             # 已占满(≥90%)或 OOM 打爆
        break                         # 完成
    if summary.starved:               # 显存平台却没占满 => 段饥饿
        multiplier *= audio_grow_factor   # 音频 ×2，重扫
    else:
        break                         # 非段饥饿(batch 已探顶等)，停止
    受 --max-audio-multiplier / --max-iterations 限制
```

**为什么段饥饿要加长音频而不是继续加 batch**：显存在增大 batch 时不再上涨，说明
可用语音段已被取空、批次填不满，再加 batch 也是空转。此时唯一能提高 GPU 占用的
办法是提供更多语音段，即把音频拼得更长。

### 5.4 两个最优档的选取

同一份扫描数据同时给出两个视角（`src/bench_core.py`）：

```python
pick_optimal(results)      # 成本最优 ⭐：RTF 最大的 ok 档（$/audio-hour 最低）
pick_saturating(results)   # 显存占满 🔥：峰值显存最大的 ok 档
```

- **成本最优 ⭐**：因为 `$/audio-hour = 时价 / RTF`，RTF 最大即单位成本最低。
- **显存占满 🔥**：报告 GPU 吃得最满那一档的 RTF / 耗时 / 成本，回答「把卡用满时
  能跑多快、多划算」。

报告（`src/report.py`）对两者都给出 RTF、中位推理耗时、峰值显存与每音频小时成本。

---

## 六、RTF 与成本的换算

`src/report.py` 中：

```python
cost_per_audio_hour = price_per_hour / r.rtf
```

推导：RTF = 每墙钟小时可转写的音频小时数 ⇒ 转写 1 小时音频需要 `1/RTF` 个
机器小时 ⇒ 乘以实例时价即每音频小时成本。**时价固定时，RTF 最大即单位成本
最低**——这正是成本最优档的目标函数。报告同时给出成本最优档 ⭐ 与显存占满档 🔥
两者的每音频小时成本。

## 七、方法的取舍与适用边界

**隐含假设**：显存随 batch 单调递增（到 OOM 为止）；显存不再随 batch 上涨即说明
语音段被取空（段饥饿）。在 batched 解码场景基本成立。

**优点**：
- 每台机器都测在「显存吃满」状态，强卡不会因音频/ batch 不足被低估；
- OOM 与显存平台都作为明确的停止边界，全程自动、无需人工试参数；
- 段饥饿自动加长音频，避免了「音频拍脑袋定长」的经验值问题。

**局限与注意**：
- 音频越长、迭代越多，单轮扫描越久。32× 起步（约 8800s）通常足以喂满到 bs≈256；
  可用 `--audio-multiplier` / `--runs` 调节耗时，用 `--max-iterations` /
  `--max-audio-multiplier` 兜底防止无限增大。
- 占满阈值 `--vram-target-frac`（默认 0.90）是「视为占满」的判定线，不追求恰好 100%
  以留出碎片余量、避免不必要的 OOM 反复。
- 本基准测的是**单进程、单音频流的离线吞吐**（分母为端到端墙钟时间、每档串行
  重复）。生产上若多进程并发喂多个音频，实际利用率与成本可能更优，本基准给出的
  是保守下界。
- 需要退回旧的「RTF 饱和提前停止、不追求占满」行为时，加 `--no-saturate`。
