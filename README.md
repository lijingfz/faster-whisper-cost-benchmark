# faster-whisper ASR 成本基准（EC2 GPU 实例性价比）

在不同 AWS EC2 GPU 实例（**G5 / G6 / G6e / G7 / G7e**）上，用 **faster-whisper large-v3**
跑离线批量转写，自动**找到该实例的最优 batch_size**、测出 **RTF（实时率）**，再结合
**On-Demand 价格**算出**每音频小时成本**，最终为每台实例生成一份成本分析报告，并可汇总成
跨实例性价比对比。

核心公式：

```
每音频小时成本 ($/audio-hour) = 实例 On-Demand 时价 ($/hour) ÷ 最优 RTF
```

RTF 是「1 小时机器墙钟时间能转写多少小时音频」。所以时价更贵但更快的实例，只要 RTF
提升幅度超过价格涨幅，单位成本反而更低——本项目用数据直接判定谁最划算。

---

## 快速开始（在目标实例上）

```bash
git clone <YOUR_REPO_URL>.git
cd faster-whisper-cost-benchmark

# 1) 一键装环境（ffmpeg + venv + torch cu128 + faster-whisper 最新版）
bash setup.sh
source .venv/bin/activate

# 2) 生成标准测试音频（把自带样本拼到约 600s）
python prepare_audio.py

# 3) 跑基准：自动检测实例/GPU，扫 batch_size，算 RTF 与成本
python benchmark.py
```

运行结束后，报告写入 `results/`：
- `results/<instance>.json`：机器可读的完整结果
- `results/<instance>.md`：该实例的成本分析报告

在每台目标实例上重复上述步骤，把各自的 `results/<instance>.json` 收集到一起后：

```bash
python aggregate_report.py --results-dir results
# => results/COST-COMPARISON.md  跨实例性价比对比表
```

---

## 工作原理

1. **实例/GPU 检测**：通过 EC2 IMDSv2 读实例类型与 region，通过 torch 读 GPU 型号、
   算力（sm_xx）、显存。
2. **batch_size 扫描**（离线吞吐目标）：依次测候选 batch_size，每档计时多次取中位数，
   算 RTF 与峰值显存。
   - 遇 CUDA OOM：标记并停止继续增大 batch。
   - 饱和提前停止：RTF 连续多档几乎不再提升时停止（`--exhaustive` 可强制扫全部）。
   - **最优 = RTF 最大的档位**（因为 $/audio-hour = 时价 ÷ RTF）。
3. **定价（三级回退）**：`--price` 手动值 → AWS Pricing API（需 boto3+凭证）→
   内置 `prices.json`。
4. **成本与报告**：算出 $/audio-hour、每 1000 音频小时成本，输出 JSON + Markdown。

---

## 为什么固定 float16 + 每机器各自最优 batch

- **float16 固定**：收敛变量、跨实例公平；且 Blackwell（sm_120，如 G7/G7e）上
  CTranslate2 **禁用了 INT8**，float16 是各代 GPU 的公共可比基准。
- **每台机器用各自最优 batch_size**：卡越强饱和点越靠后（L4 甜点约 bs=8~16，
  Blackwell 可能到 bs=32~48）。固定同一 batch 会系统性低估强卡，因此必须各自寻优。

> ⚠️ Blackwell 提示：CTranslate2 对 sm_120 调优尚不成熟，厂商标称算力不一定全部兑现为
> RTF，务必以实测为准。项目依赖最新 faster-whisper / CTranslate2 以获得 Blackwell 支持。

---

## 常用参数

```bash
python benchmark.py \
  --batch-sizes 1,4,8,16,24,32,48,64,96,128 \  # 候选 batch（默认已覆盖 Blackwell 上探区间）
  --runs 3 \                          # 每档计时次数取中位数
  --price 2.52 \                      # 手动指定 On-Demand 时价（最高优先级）
  --exhaustive \                      # 不因饱和提前停止，扫全部候选
  --audio audio/benchmark-audio.mp3   # 自定义音频
```

## 目录结构

```
benchmark.py           主入口：检测→扫batch→算RTF→算成本→出报告
aggregate_report.py    汇总多实例结果成对比报告
prepare_audio.py       生成标准测试音频
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
