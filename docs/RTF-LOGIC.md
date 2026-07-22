# RTF 计算与最优 batch_size 寻优逻辑详解

本文详细说明本项目中 RTF（Real-Time Factor，实时率）的计算方式，以及"如何获得最佳
RTF"的完整搜索逻辑。相关实现集中在 `src/bench_core.py` 与 `benchmark.py`。

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
得到 `audio_duration_s`。整个基准过程中它是固定常数（默认约 600 秒）。

**为什么音频要拼接到约 600 秒**（见 `prepare_audio.py`）：
`BatchedInferencePipeline` 先用 VAD 把长音频切成语音段，再按 batch_size 并行送入
GPU 解码。音频太短则切出的段数少于 batch_size，大 batch「填不满」，测不出真实
吞吐。同时 RTF 是比值，与音频绝对长度无关——只要所有实例用同一份音频，跨机
对比即公平。

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
| 每档跑 `--runs` 次取**中位数**（默认 3 次） | `sweep_batch_sizes()` | 中位数对偶发抖动比平均数更稳健 |
| 固定语种 `--language en` | `benchmark.py` | 跳过每次运行的语种自动检测，消除该部分耗时波动 |

测量期间另有 `GpuMemSampler` 后台线程以 50ms 间隔采样显存峰值，运行在独立
Python 线程上，对 GPU 计时的干扰可忽略。

---

## 五、如何获得最佳 RTF：batch_size 一维扫描

### 5.1 为什么 batch_size 是唯一的搜索变量

RTF 随 batch_size 变化的物理机制：pipeline 每次取 batch_size 个 VAD 语音段并行
送入 GPU 解码。batch 越大，GPU 并行单元填得越满、单位开销摊得越薄，RTF 上升；
直到 GPU 算力饱和，曲线走平；再往上只增加显存占用甚至 OOM。因此
**RTF-batch 曲线是「先陡升、后平台」的单峰形状，最佳 RTF 就在平台起点附近**。

其他影响 RTF 的变量都被刻意固定，以保证搜索只有一维：

- 模型固定 `large-v3`，精度固定 `float16`（跨实例公平，且 Blackwell 上 INT8 不可用）
- 语种固定 `en`（跳过检测）
- 音频固定为同一份约 600 秒的拼接文件

### 5.2 扫描算法（`sweep_batch_sizes()`）

按升序逐档测试候选列表（默认见 `benchmark.py` 的 `DEFAULT_BATCH_SIZES`），
对每一档：

**测量**：跑 `runs` 次，取中位耗时，算出该档 RTF 与峰值显存。

**异常分流**：
- 抛出 **CUDA OOM** → 标记该档 `oom`，**终止整个扫描**。依据：显存占用随
  batch 单调递增，更大的档必然也 OOM。
- 抛出**其他错误** → 标记 `error`，跳过该档继续测下一档（偶发错误不应终结
  整个基准）。

**饱和提前停止**（高效寻优的核心）：

```python
if res.rtf and res.rtf > best_rtf * (1 + early_stop_epsilon):
    best_rtf = max(best_rtf, res.rtf)
    stagnant = 0                # 有显著提升，重置计数
else:
    best_rtf = max(best_rtf, res.rtf or 0.0)
    stagnant += 1               # 无显著提升，累计一次「停滞」
    if stagnant >= early_stop_patience:
        break                   # 连续停滞达到耐心值，判定饱和
```

两个设计细节：

1. **比较基准是「历史最佳 RTF」，不是「上一档」**。新档必须比迄今最好成绩再高出
   `epsilon`（默认 2%）才算「有提升」。这能正确处理平台期曲线小幅抖动的情况——
   比上一档高一点但没超过历史最佳，仍算停滞。
2. **`patience=2` 容忍一次偶然停滞**。单档停滞可能只是测量噪声；连续两档都挤
   不出 2% 提升，才判定真的到了平台。`--exhaustive` 会将 patience 设为极大值，
   等效关闭此机制、强制扫全部候选。

### 5.3 最终选取（`pick_optimal()`）

```python
ok = [r for r in results if r.status == "ok" and r.rtf]
return max(ok, key=lambda r: r.rtf)
```

在所有**成功**档位里取 RTF 最大者。注意：**最佳档不一定是最后测的那档**——
提前停止要求多测 2 个停滞档才收手，峰值往往出现在倒数第三档；`pick_optimal`
对全体结果取 max，自然选回真正的峰值。OOM / error 档位一律排除，即 OOM 是
**安全边界**而非搜索目标。

### 5.4 示例：典型 L4（g6 实例）走查

假设实测曲线如下（epsilon=2%，patience=2）：

| batch_size | RTF | 判断 |
|---:|---:|---|
| 1 | 6x | 提升，best=6 |
| 4 | 20x | 20 > 6×1.02，提升，best=20 |
| 8 | 34x | 提升，best=34 |
| 16 | 41x | 提升，best=41 |
| 24 | 41.5x | 41.5 < 41×1.02=41.8，**停滞 1** |
| 32 | 40x | **停滞 2 → 触发饱和停止** |

更大的档不再运行。`pick_optimal` 从 6 个结果取 max →
**最优 batch_size=24，RTF=41.5x**（虽然扫描在 32 停止）。更强的卡（如 Blackwell）
饱和点靠后，扫描会自然走到更大的候选档。

---

## 六、RTF 与成本的换算

`src/report.py` 中：

```python
cost_per_audio_hour = price_per_hour / optimal.rtf
```

推导：RTF = 每墙钟小时可转写的音频小时数 ⇒ 转写 1 小时音频需要 `1/RTF` 个
机器小时 ⇒ 乘以实例时价即每音频小时成本。**时价固定时，RTF 最大即单位成本
最低**——这正是 batch 寻优的目标函数。

## 七、方法的取舍与适用边界

**隐含假设**：RTF-batch 曲线单峰（升→平/降），在 GPU batching 场景基本成立。

**优点**：
- 无需扫全列表，弱卡上通常测 5~6 档即结束；
- OOM 只作为边界处理，正常情况下远未 OOM 就已收敛。

**局限**：
- 分辨率受限于候选列表步长：若真实峰值在两档之间（如 16 与 24 之间的 20），
  只能取到两者中较好者。对成本估算而言误差 <2%，可接受；
- 本基准测的是**单进程、单音频流的离线吞吐**（分母为端到端墙钟时间、每档串行
  重复）。生产上若多进程并发喂多个音频，实际机器利用率与成本可能更优，
  本基准给出的是保守下界。
