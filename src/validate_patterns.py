"""
Check that the planted patterns can be recovered from the generated data.

This is a quick pandas check before the dbt models exist. It applies the
project's KPI rules:

  * Grain: schedule line (EKET).
  * Goods receipts (EKBE) are allocated to an item's schedule lines in date
    order (first in, first out).
  * Strict on-time rule: a schedule line is on time only if its FULL quantity
    has been received on or before its promised date (EINDT).
  * Only lines due on or before the as-of date count.

Usage:
    python src/validate_patterns.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")


def load():
    ekko = pd.read_csv(RAW / "ekko.csv", dtype=str)
    eket = pd.read_csv(RAW / "eket.csv", dtype={"EBELN": str, "EBELP": str, "ETENR": str},
                       parse_dates=["EINDT"])
    ekbe = pd.read_csv(RAW / "ekbe.csv", dtype={"EBELN": str, "EBELP": str},
                       parse_dates=["BUDAT"])
    truth = json.loads((RAW / "planted_patterns.json").read_text())
    return ekko, eket, ekbe, truth


def allocate(eket: pd.DataFrame, ekbe: pd.DataFrame) -> pd.DataFrame:
    """For each schedule line, find when it was fully received (FIFO) and how much arrived."""
    out = []
    receipts = {k: g.sort_values("BUDAT") for k, g in ekbe.groupby(["EBELN", "EBELP"])}
    for key, lines in eket.sort_values("EINDT").groupby(["EBELN", "EBELP"]):
        rec = receipts.get(key)
        cum_needed = 0
        for line in lines.itertuples(index=False):
            cum_needed += line.MENGE
            full_date, received = pd.NaT, 0
            if rec is not None:
                cum = rec["MENGE"].cumsum()
                prev_needed = cum_needed - line.MENGE
                received = int(min(max(cum.iloc[-1] - prev_needed, 0), line.MENGE))
                hit = rec.loc[cum >= cum_needed, "BUDAT"]
                if not hit.empty:
                    full_date = hit.iloc[0]
            out.append({"EBELN": key[0], "EBELP": key[1], "ETENR": line.ETENR,
                        "EINDT": line.EINDT, "MENGE": line.MENGE,
                        "RECEIVED": received, "FULL_DATE": full_date})
    return pd.DataFrame(out)


def main() -> None:
    ekko, eket, ekbe, truth = load()
    as_of = pd.Timestamp(truth["as_of"])

    lines = allocate(eket, ekbe).merge(ekko[["EBELN", "LIFNR"]], on="EBELN")
    due = lines[lines["EINDT"] <= as_of].copy()
    due["ON_TIME"] = due["FULL_DATE"].notna() & (due["FULL_DATE"] <= due["EINDT"])
    due["OPEN_OVERDUE"] = due["FULL_DATE"].isna()

    # Quantity accuracy: only lines that were delivered (at least one receipt)
    # and are long past due, so in-transit remainders don't count against suppliers
    settled = due[(due["EINDT"] <= as_of - pd.Timedelta(days=60)) & (due["RECEIVED"] > 0)]
    kpis = (due.groupby("LIFNR")
              .agg(lines_due=("ON_TIME", "size"),
                   on_time_rate=("ON_TIME", "mean"),
                   open_overdue=("OPEN_OVERDUE", "sum"))
              .join(settled.groupby("LIFNR")
                           .apply(lambda g: g["RECEIVED"].sum() / g["MENGE"].sum())
                           .rename("qty_accuracy"))
              .sort_values("on_time_rate"))

    pd.set_option("display.width", 120)
    print(kpis.round(3).to_string())
    print()

    late = set(truth["chronically_late_suppliers"])
    partial = truth["partial_shipper"]
    bottom4 = set(kpis.index[:4])
    lowest_qty = kpis["qty_accuracy"].idxmin()

    checks = {
        "3 planted late suppliers are among the 4 lowest on-time rates": late <= bottom4,
        "planted partial shipper is among the 4 lowest on-time rates": partial in bottom4,
        "planted partial shipper has the lowest quantity accuracy": lowest_qty == partial,
    }
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    overall = due["ON_TIME"].mean()
    print(f"\nOverall on-time rate (strict rule): {overall:.1%} across {len(due)} due schedule lines")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
