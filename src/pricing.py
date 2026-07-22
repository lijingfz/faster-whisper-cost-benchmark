"""On-Demand 价格获取，三级回退（优先级从高到低）：

1. CLI 显式传入 --price
2. AWS Pricing API（需 boto3 + 凭证 + 网络；Pricing API 端点在 us-east-1 / ap-south-1）
3. 内置 prices.json（离线回退，可手动更新）

返回 (price_per_hour, source)。全部失败返回 (None, "unavailable")。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

# region code -> Pricing API 使用的 location 全称
_REGION_TO_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-north-1": "EU (Stockholm)",
    "ca-central-1": "Canada (Central)",
    "sa-east-1": "South America (Sao Paulo)",
}


def _from_pricing_api(instance_type: str, region: str) -> Optional[float]:
    """通过 AWS Pricing API 查 Linux/Shared/On-Demand 单价。失败返回 None。"""
    location = _REGION_TO_LOCATION.get(region)
    if not location:
        return None
    try:
        import boto3
    except ImportError:
        return None
    try:
        # Pricing API 仅在 us-east-1 / ap-south-1 有端点
        client = boto3.client("pricing", region_name="us-east-1")
        resp = client.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            MaxResults=1,
        )
        for price_str in resp.get("PriceList", []):
            data = json.loads(price_str)
            terms = data.get("terms", {}).get("OnDemand", {})
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    usd = dim.get("pricePerUnit", {}).get("USD")
                    if usd is not None:
                        val = float(usd)
                        if val > 0:
                            return val
    except Exception:
        return None
    return None


def _from_prices_json(instance_type: str, region: str, prices_file: str
                      ) -> Optional[float]:
    """从内置 prices.json 读取。优先精确匹配 region，否则用 default_region。"""
    if not os.path.exists(prices_file):
        return None
    try:
        with open(prices_file) as f:
            data = json.load(f)
    except Exception:
        return None
    entry = data.get("instances", {}).get(instance_type)
    if not entry:
        return None
    by_region = entry.get("on_demand_usd_per_hour", {})
    if region in by_region:
        return float(by_region[region])
    default_region = data.get("default_region")
    if default_region and default_region in by_region:
        return float(by_region[default_region])
    # 退而求其次：取任意一个已知区域的价格
    if by_region:
        return float(next(iter(by_region.values())))
    return None


def get_on_demand_price(
    instance_type: str,
    region: str,
    prices_file: str,
    cli_price: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """按三级优先级返回 (每小时美元价, 来源标签)。"""
    if cli_price is not None:
        return cli_price, "cli_override"
    api_price = _from_pricing_api(instance_type, region)
    if api_price is not None:
        return api_price, "aws_pricing_api"
    json_price = _from_prices_json(instance_type, region, prices_file)
    if json_price is not None:
        return json_price, "prices_json_fallback"
    return None, "unavailable"
