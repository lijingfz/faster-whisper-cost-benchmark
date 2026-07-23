# faster-whisper ASR 成本基准（EC2 GPU 实例性价比）

在不同 AWS EC2 GPU 实例（**G5 / G6 / G6e / G7 / G7e**）上，用 **faster-whisper large-v3**
跑离线批量转写。全自动把 GPU **显存推到极限**（占满 ~90% 或直接 OOM 打爆），在此基础上
测出 **RTF（实时率）**、**推理耗时**，再结合 **On-Demand 价格**算出**每音频小时成本**，
为每台实例生成成本分析报告，并可汇总成跨实例性价比对比。

核心公式：

```
每音频小时成本 ($/audio-hour) = 实例 On-Demand 时价 ($/hour) ÷ RTF
```

RTF 是「1 小时机器墙钟时间能转写多少小时音频」。所以时价更贵但更快的实例，只要 RTF
提升幅度超过价格涨幅，单位成本反而更低——本项目用数据直接判定谁最划算。

> **为什么要"占满显存"**：`large-v3` 按 30s 窗口解码，`BatchedInferencePipeline` 先用 VAD
> 把音频切成语音段再按 batch 并行。**音频太短时切出的段数少于 batch_size，大 batch 填不满**，
> 显存和 RTF 会在远未到硬件极限时就"假性走平"，从而**系统性低估强卡**。本项目通过自适应
> 增大音频，确保测到的是 GPU 真正吃满时的吞吐。

---

## 快速开始（在目标实例上）

```bash
git clone <YOUR_REPO_URL>.git
cd faster-whisper-cost-benchmark

# 1) 一键装环境（ffmpeg + venv + torch cu128 + faster-whisper 最新版）
bash setup.sh
source .venv/bin/activate

# 2) 跑基准：全自动。自动生成音频→占满显存→算 RTF/耗时/成本
python benchmark.py
```

`benchmark.py` 会**自动生成测试音频**（默认 base 样本的 20 倍），无需手动跑
`prepare_audio.py`。运行结束后报告写入 `results/`：
- `results/<instance>.json`：机器可读的完整结果（含显存利用率、饱和状态）
- `results/<instance>.md`：该实例的成本分析报告

在每台目标实例上重复后，把各自的 `results/<instance>.json` 收集到一起：

```bash
python aggregate_report.py --results-dir results
# => results/COST-COMPARISON.md  跨实例性价比对比表（含显存利用率列）
```

---

## 工作原理

1. **实例/GPU 检测**：通过 EC2 IMDSv2 读实例类型与 region，通过 torch 读 GPU 型号、
   算力（sm_xx）、显存总量。
2. **自适应显存占满闭环**（全自动）：
   1. 从 base 音频的 **20 倍**起步生成测试音频（`--audio-multiplier`）。
   2. 扫描 batch_size 并**一路向上翻倍探顶**：遇 **OOM**（显存打爆）或显存达到
      **目标阈值 90%**（`--vram-target-frac`）即视为占满。
   3. 若显存在增大 batch 时**不再上涨却仍未占满**（判定为**语音段不足/段饥饿**），
      自动把音频**倍数 ×2**（`--audio-grow-factor`）后重扫，直到占满或触及上限
      （`--max-audio-multiplier` / `--max-iterations`）。
3. **计算指标**：在占满状态下，对每个成功档位算 RTF、中位推理耗时、峰值显存。
   - **成本最优档 ⭐**：RTF 最大（$/audio-hour 最低）。
   - **显存占满档 🔥**：峰值显存最大（GPU 吃得最满）。
4. **定价（三级回退）**：`--price` 手动值 → AWS Pricing API（需 boto3+凭证）→
   内置 `prices.json`。
5. **成本与报告**：算出 $/audio-hour、每 1000 音频小时成本，输出 JSON + Markdown。

> RTF 的精确计算方式、显存占满与段饥饿的自适应算法（OOM 边界、显存平台判定、
> 音频自动增大等）详见 [docs/RTF-LOGIC.md](docs/RTF-LOGIC.md)。

---

## 为什么固定 float16 + 每机器各自占满显存

- **float16 固定**：收敛变量、跨实例公平；且 Blackwell（sm_120，如 G7/G7e）上
  CTranslate2 **禁用了 INT8**，float16 是各代 GPU 的公共可比基准。
- **每台机器各自把显存占满**：卡越强、显存越大，饱和点越靠后。若音频/ batch 不足以
  喂满强卡，会系统性低估它。自适应占满保证每台机器都测在"吃满"状态，跨机对比才公平。

> ⚠️ Blackwell 提示：CTranslate2 对 sm_120 调优尚不成熟，厂商标称算力不一定全部兑现为
> RTF，务必以实测为准。项目依赖最新 faster-whisper / CTranslate2 以获得 Blackwell 支持。

> ⏱️ 运行时长提示：音频越长、迭代越多，单轮扫描越久。20× 起步（约 5500s 音频）通常
> 足以喂满到 bs≈256；如需更快可调小 `--audio-multiplier` 或 `--runs`，如遇段饥饿会自动增大。

---

## 常用参数

```bash
python benchmark.py \
  --audio-multiplier 20 \      # 起始音频 = base 的多少倍（默认 20）
  --vram-target-frac 0.90 \    # 显存利用率达到该比例即视为"占满"（默认 0.90）
  --audio-grow-factor 2.0 \    # 段饥饿时音频倍数的增长系数（默认 2.0）
  --max-audio-multiplier 1024 \# 音频倍数安全上限
  --max-batch 1024 \           # batch 上探安全上限
  --max-iterations 6 \         # 自适应增大音频的最大轮数
  --batch-sizes 1,4,8,16,32,48,64,80,96 \  # 起始候选 batch（saturate 模式会自动继续翻倍探顶）
  --runs 1 \                   # 每档计时次数取中位数（默认 1）
  --price 2.52 \               # 手动指定 On-Demand 时价（最高优先级）
  --no-saturate \              # 退回旧模式：RTF 饱和提前停止，不追求占满显存
  --audio audio/my.mp3         # 直接指定音频（跳过自动生成与自适应增大）
```

单独生成/预生成测试音频（可选，`benchmark.py` 会自动做）：

```bash
python prepare_audio.py --multiplier 32     # 拼成 base 的 32 倍
python prepare_audio.py --target 900        # 或按目标时长(秒)拼接
```

## 目录结构

```
benchmark.py           主入口：检测→自适应占满显存→算RTF/耗时→算成本→出报告
aggregate_report.py    汇总多实例结果成对比报告（含显存利用率列）
prepare_audio.py       生成/拼接标准测试音频（build_audio 供 benchmark 调用）
prices.json            On-Demand 价格回退表（可更新）
setup.sh               一键环境搭建
requirements.txt       依赖（torch 单独按 CUDA wheel 装）
src/                   gpu_info / pricing / bench_core / report
audio/base-sample.mp3  自带基准音频样本
results/               每实例报告与汇总输出
```

## 定价数据说明

`prices.json` 的价格有 `last_verified` 日期，会随时间/区域变化。运行环境若配置了 AWS
凭证，会优先用 **Pricing API 实时价格**；否则用本表回退。也可随时用 `--price` 覆盖。
请以 [AWS EC2 On-Demand 定价页](https://aws.amazon.com/ec2/pricing/on-demand/) 为准。

## 许可

MIT，见 `LICENSE`。自带 `audio/base-sample.mp3` 为演示用合成通话音频；你可替换为自己的
测试音频（跨实例对比时确保所有机器用同一份）。
