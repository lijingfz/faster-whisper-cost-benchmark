#!/usr/bin/env bash
# 一键环境搭建：ffmpeg + venv + torch(CUDA 12.8 wheel) + faster-whisper
# 适用于 G5/G6/G6e/G7/G7e（含 Blackwell sm_100/sm_120，需最新 CTranslate2）。
set -euo pipefail

echo "==> 安装 ffmpeg（音频解码所需）"
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq ffmpeg python3-venv
fi

echo "==> 创建虚拟环境 .venv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

echo "==> 安装 torch（CUDA 12.8 wheel，兼容 Blackwell）"
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "==> 安装 faster-whisper（最新版）+ 依赖"
pip install -r requirements.txt

echo "==> 校验"
python - <<'PY'
import torch, faster_whisper, ctranslate2
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0),
          "sm_%d%d" % torch.cuda.get_device_capability(0))
print("faster_whisper", faster_whisper.__version__)
print("ctranslate2", ctranslate2.__version__)
PY

echo "==> 完成。接下来："
echo "    source .venv/bin/activate"
echo "    python prepare_audio.py      # 生成测试音频"
echo "    python benchmark.py          # 跑基准并生成成本报告"
