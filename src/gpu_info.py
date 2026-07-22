"""实例与 GPU 信息检测。

- EC2 实例类型 / region：优先 IMDSv2（EC2 元数据服务），失败则回退到环境变量或 "unknown"。
- GPU：通过 torch 读取名称、compute capability、显存总量。

所有网络调用都带短超时，非 EC2 环境不会卡住。
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

_IMDS_BASE = "http://169.254.169.254/latest"
_IMDS_TIMEOUT = 1.0  # 秒，非 EC2 环境快速失败


def _imds_token() -> Optional[str]:
    try:
        req = urllib.request.Request(
            f"{_IMDS_BASE}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT) as r:
            return r.read().decode()
    except Exception:
        return None


def _imds_get(path: str, token: Optional[str]) -> Optional[str]:
    try:
        headers = {"X-aws-ec2-metadata-token": token} if token else {}
        req = urllib.request.Request(f"{_IMDS_BASE}/meta-data/{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT) as r:
            return r.read().decode().strip()
    except Exception:
        return None


def get_instance_type(override: Optional[str] = None) -> str:
    """返回 EC2 实例类型，如 'g6.2xlarge'。检测不到返回 'unknown'。"""
    if override:
        return override
    if os.environ.get("INSTANCE_TYPE"):
        return os.environ["INSTANCE_TYPE"]
    token = _imds_token()
    return _imds_get("instance-type", token) or "unknown"


def get_region(override: Optional[str] = None) -> str:
    """返回 region，如 'us-east-1'。检测不到返回 'unknown'。"""
    if override:
        return override
    if os.environ.get("AWS_REGION"):
        return os.environ["AWS_REGION"]
    token = _imds_token()
    return _imds_get("placement/region", token) or "unknown"


@dataclass
class GpuInfo:
    name: str
    compute_capability: str  # 如 "sm_89"
    total_vram_gib: float
    device_count: int


def get_gpu_info() -> GpuInfo:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA GPU。本基准需要 NVIDIA GPU。")
    idx = 0
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    _, total = torch.cuda.mem_get_info(idx)
    return GpuInfo(
        name=name,
        compute_capability=f"sm_{major}{minor}",
        total_vram_gib=round(total / (1024 ** 3), 2),
        device_count=torch.cuda.device_count(),
    )


def as_dict(info: GpuInfo) -> dict:
    return asdict(info)
