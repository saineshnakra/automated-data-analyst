"""Deterministic demo dataset with enough structure for a useful dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(seed: int = 17, rows: int = 1_800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    products = np.array(["Core", "Growth", "Enterprise", "Starter"])
    product_prices = {"Core": 129.0, "Growth": 229.0, "Enterprise": 799.0, "Starter": 49.0}
    regions = np.array(["West", "Northeast", "South", "Midwest"])
    channels = np.array(["Direct", "Partner", "Self-serve"])

    order_date = rng.choice(dates, size=rows)
    product = rng.choice(products, size=rows, p=[0.40, 0.27, 0.11, 0.22])
    region = rng.choice(regions, size=rows, p=[0.31, 0.27, 0.25, 0.17])
    channel = rng.choice(channels, size=rows, p=[0.45, 0.22, 0.33])
    units = rng.integers(1, 6, size=rows)

    day_index = (pd.Series(order_date) - pd.Timestamp("2024-01-01")).dt.days.to_numpy()
    growth = 1 + day_index / day_index.max() * 0.28
    seasonal = 1 + 0.13 * np.sin(2 * np.pi * day_index / 365)
    west_lift = np.where(region == "West", 1.10, 1.0)
    enterprise_lift = np.where(product == "Enterprise", 1.18, 1.0)
    base_price = np.array([product_prices[item] for item in product])
    revenue = base_price * units * growth * seasonal * west_lift * enterprise_lift
    revenue *= rng.normal(1.0, 0.09, size=rows)
    cost_ratio = np.where(product == "Enterprise", 0.58, 0.67)
    profit = revenue * (1 - cost_ratio) - rng.uniform(4, 18, size=rows)

    return pd.DataFrame(
        {
            "Order ID": [f"ORD-{100_000 + index}" for index in range(rows)],
            "Order Date": pd.to_datetime(order_date),
            "Product": product,
            "Region": region,
            "Channel": channel,
            "Units": units,
            "Revenue": revenue.round(2),
            "Profit": profit.round(2),
        }
    ).sort_values("Order Date", ignore_index=True)
