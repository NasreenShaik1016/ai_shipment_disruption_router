"""Test-case CSV loading and conversion to DisruptionEvent."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from logistics_agent.schemas import DisruptionEvent

REQUIRED_COLUMNS = [
    "Order Id", "Days for shipping (real)", "Days for shipment (scheduled)",
    "Delivery Status", "Late_delivery_risk", "Shipping Mode", "Customer Segment",
    "Category Name", "Sales", "Order Region", "Order City",
    "decision_expected", "expected_policies",
]

CASE_LABELS = {
    "TC001": "On-Time Monitoring",
    "TC002": "Standard Reroute (Corporate Late Delivery)",
    "TC003": "Critical Escalation (Canceled Shipment)",
    "TC004": "High-Value Expedite (Electronics)",
    "TC005": "Minor Delay Hold",
}


def load_test_cases(path: Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Test case file not found: {path}")
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Test case file is missing columns: {missing}")
    return df


def row_to_event(row: pd.Series) -> DisruptionEvent:
    return DisruptionEvent(
        order_id=str(row["Order Id"]),
        delivery_status=row["Delivery Status"],
        shipping_mode=row["Shipping Mode"],
        days_scheduled=int(row["Days for shipment (scheduled)"]),
        days_real=int(row["Days for shipping (real)"]),
        late_risk=int(row["Late_delivery_risk"]),
        customer_segment=row["Customer Segment"],
        sales=float(row["Sales"]),
        category=row["Category Name"],
        region=row["Order Region"],
        city=str(row["Order City"]),
    )


def build_cases(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    return [(CASE_LABELS.get(str(r["Order Id"]), str(r["Order Id"])), r) for _, r in df.iterrows()]
