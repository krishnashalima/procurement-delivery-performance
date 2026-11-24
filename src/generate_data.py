"""
Generate synthetic, SAP-shaped purchasing data for the procurement
delivery-performance project.

The tables mimic simplified versions of SAP ERP / S/4HANA purchasing tables:

    LFA1  supplier (vendor) master
    MARA  material master (material text included here for simplicity;
          in SAP it lives in MAKT)
    EKKO  purchase order header
    EKPO  purchase order item
    EKET  schedule lines (promised delivery dates and quantities)
    EKBE  purchase order history, goods receipts only (VGABE = '1')

All data is synthetic. Known patterns are planted on purpose so the
report built on top can be validated against them (see
data/raw/planted_patterns.json).

Usage:
    python src/generate_data.py                 # default settings, seed 42
    python src/generate_data.py --seed 7        # different random data
    python src/generate_data.py --inject-dq     # add a few deliberate data-quality
                                                # errors for testing the dbt tests
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

START_DATE = date(2025, 1, 1)     # first PO date
END_DATE = date(2026, 10, 31)     # last PO date
AS_OF = date(2026, 11, 15)        # fixed "today", so results are reproducible

N_SUPPLIERS = 25
N_MATERIALS = 60
N_POS = 3000

COMPANY_CODE = "1000"
PURCH_ORG = "1000"
PLANTS = ["1010", "1020"]
PURCH_GROUPS = ["001", "002", "003"]

# Material groups with a typical lead time (days from PO date to promised date)
MATERIAL_GROUPS = {
    "MECH": ("Mechanical parts", 21),
    "ELEC": ("Electrical components", 28),
    "PIPE": ("Pipes and fittings", 35),
    "CONS": ("Consumables", 10),
    "SAFE": ("Safety equipment", 14),
}

COUNTRIES = [("DE", "Berlin"), ("DE", "Hamburg"), ("DE", "Munich"),
             ("PL", "Poznan"), ("CZ", "Brno"), ("NL", "Rotterdam"),
             ("IT", "Milan"), ("IN", "Bengaluru")]

# Supplier behaviour profiles
PROFILE_NORMAL = "normal"
PROFILE_LATE = "chronically_late"
PROFILE_PARTIAL = "partial_shipper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def daterange_days(start: date, end: date) -> int:
    return (end - start).days


def seasonal_extra_delay(d: date, rng: np.random.Generator) -> int:
    """Extra delay in August (summer holidays) and December (year-end)."""
    if d.month in (8, 12) and rng.random() < 0.35:
        return int(rng.integers(2, 8))
    return 0


# ---------------------------------------------------------------------------
# Master data
# ---------------------------------------------------------------------------

def make_suppliers(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    ids = [f"{100001 + i:010d}" for i in range(N_SUPPLIERS)]
    rows = []
    for i, lifnr in enumerate(ids):
        land, city = COUNTRIES[i % len(COUNTRIES)]
        rows.append({
            "LIFNR": lifnr,                       # supplier number
            "NAME1": f"Supplier {i + 1:02d} GmbH" if land == "DE" else f"Supplier {i + 1:02d} Ltd",
            "LAND1": land,                        # country key
            "ORT01": city,                        # city
            "KTOKK": "LIEF",                      # account group
        })
    suppliers = pd.DataFrame(rows)

    # Order volume per supplier: some suppliers get more orders (realistic concentration)
    weights = rng.dirichlet(np.ones(len(ids)) * 1.5)

    # Plant the behaviour: 3 chronically late suppliers and 1 partial shipper,
    # picked from the busier half so each has enough lines to measure
    busy = [ids[i] for i in np.argsort(weights)[::-1][: len(ids) // 2]]
    shuffled = rng.permutation(busy)
    late = sorted(shuffled[:3].tolist())
    partial = str(shuffled[3])
    profiles = {s: PROFILE_NORMAL for s in ids}
    for s in late:
        profiles[s] = PROFILE_LATE
    profiles[partial] = PROFILE_PARTIAL
    return suppliers, profiles, weights


def make_materials(rng: np.random.Generator) -> pd.DataFrame:
    groups = list(MATERIAL_GROUPS)
    rows = []
    for i in range(N_MATERIALS):
        g = groups[i % len(groups)]
        rows.append({
            "MATNR": f"MAT-{10001 + i}",          # material number
            "MAKTX": f"{MATERIAL_GROUPS[g][0]} item {i + 1:02d}",
            "MATKL": g,                           # material group
            "MEINS": "EA" if g != "PIPE" else "M",  # base unit of measure
            "BASE_PRICE": round(float(rng.uniform(5, 400)), 2),  # helper, not an SAP field
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transactional data
# ---------------------------------------------------------------------------

def make_purchase_orders(rng, suppliers, materials, weights):
    span = daterange_days(START_DATE, END_DATE)
    ekko, ekpo, eket = [], [], []
    lifnrs = suppliers["LIFNR"].tolist()

    for n in range(N_POS):
        ebeln = f"{4500000001 + n}"
        bedat = START_DATE + timedelta(days=int(rng.integers(0, span + 1)))
        lifnr = str(rng.choice(lifnrs, p=weights))
        ekko.append({
            "EBELN": ebeln,                # PO number
            "BUKRS": COMPANY_CODE,         # company code
            "BSART": "NB",                 # PO type: standard
            "LIFNR": lifnr,                # supplier
            "BEDAT": bedat.isoformat(),    # PO date
            "EKORG": PURCH_ORG,            # purchasing organisation
            "EKGRP": str(rng.choice(PURCH_GROUPS)),  # purchasing group
            "WAERS": "EUR",                # currency
        })

        n_items = int(rng.integers(1, 6))
        mats = materials.sample(n=n_items, random_state=int(rng.integers(0, 1_000_000)))
        for k, (_, m) in enumerate(mats.iterrows(), start=1):
            ebelp = f"{k * 10:05d}"
            qty = int(rng.integers(5, 500))
            price = round(m["BASE_PRICE"] * float(rng.uniform(0.9, 1.1)), 2)
            ekpo.append({
                "EBELN": ebeln, "EBELP": ebelp,
                "MATNR": m["MATNR"], "WERKS": str(rng.choice(PLANTS)),
                "MENGE": qty, "MEINS": m["MEINS"],
                "NETPR": price, "PEINH": 1,
            })

            lead = MATERIAL_GROUPS[m["MATKL"]][1] + int(rng.integers(-3, 8))
            first_date = bedat + timedelta(days=lead)
            # About 10 % of items are delivered on two scheduled dates
            if rng.random() < 0.10 and qty >= 20:
                q1 = int(qty * rng.uniform(0.4, 0.6))
                lines = [(first_date, q1),
                         (first_date + timedelta(days=int(rng.integers(14, 31))), qty - q1)]
            else:
                lines = [(first_date, qty)]
            for etenr, (eindt, sq) in enumerate(lines, start=1):
                eket.append({
                    "EBELN": ebeln, "EBELP": ebelp, "ETENR": f"{etenr:04d}",
                    "EINDT": eindt.isoformat(),   # promised delivery date
                    "MENGE": sq,                  # scheduled quantity
                })
    return pd.DataFrame(ekko), pd.DataFrame(ekpo), pd.DataFrame(eket)


def receipts_for_line(rng, profile, eindt: date, qty: int):
    """Return a list of (posting_date, quantity) goods receipts for one schedule line."""
    # How late is the (first) delivery?
    if profile == PROFILE_LATE and rng.random() < 0.35:
        delay = int(rng.integers(5, 21))
    elif profile != PROFILE_LATE and rng.random() < 0.08:
        delay = int(rng.integers(1, 5))
    else:
        delay = int(rng.integers(-3, 1))          # on time or a little early
    delay += seasonal_extra_delay(eindt, rng)

    # A small share of recently due lines is still outstanding
    recent = (AS_OF - eindt).days < 120
    if recent and rng.random() < (0.25 if profile == PROFILE_LATE else 0.08):
        delay = 400                               # lands after AS_OF, so not posted

    first = eindt + timedelta(days=delay)

    if profile == PROFILE_PARTIAL and rng.random() < 0.40 and qty >= 4:
        q1 = max(1, int(qty * rng.uniform(0.5, 0.8)))
        rest = qty - q1
        if rng.random() < 0.5:                    # short-shipped: remainder never fully arrives
            rest = int(rest * rng.uniform(0.6, 0.9))
        second = first + timedelta(days=int(rng.integers(3, 16)))
        out = [(first, q1)]
        if rest > 0:
            out.append((second, rest))
        return out
    return [(first, qty)]


def make_goods_receipts(rng, ekko, eket, profiles):
    lifnr_by_po = dict(zip(ekko["EBELN"], ekko["LIFNR"]))
    rows = []
    for line in eket.itertuples(index=False):
        profile = profiles[lifnr_by_po[line.EBELN]]
        eindt = date.fromisoformat(line.EINDT)
        for budat, q in receipts_for_line(rng, profile, eindt, int(line.MENGE)):
            if budat > AS_OF:                     # not happened yet as of "today"
                continue
            rows.append({"EBELN": line.EBELN, "EBELP": line.EBELP,
                         "BUDAT": budat.isoformat(), "MENGE": q})

    ekbe = pd.DataFrame(rows).sort_values(["BUDAT", "EBELN", "EBELP"]).reset_index(drop=True)
    ekbe.insert(2, "VGABE", "1")                  # transaction type: goods receipt
    ekbe.insert(3, "GJAHR", ekbe["BUDAT"].str[:4])  # fiscal year (calendar year here)
    ekbe.insert(4, "BELNR", [f"{5000000001 + i}" for i in range(len(ekbe))])  # material document
    ekbe.insert(5, "BUZEI", "0001")               # document line
    ekbe["BWART"] = "101"                         # movement type: goods receipt for PO
    ekbe["SHKZG"] = "S"                           # debit
    return ekbe


def inject_data_quality_errors(rng, ekko, ekbe):
    """Add a handful of deliberate errors so the data-quality tests have something to catch."""
    ekbe = ekbe.copy()
    idx = rng.choice(ekbe.index, size=3, replace=False)
    po_date = dict(zip(ekko["EBELN"], ekko["BEDAT"]))
    # 1. goods receipt dated before its PO date
    ekbe.loc[idx[0], "BUDAT"] = (date.fromisoformat(po_date[ekbe.loc[idx[0], "EBELN"]])
                                 - timedelta(days=5)).isoformat()
    # 2. negative quantity
    ekbe.loc[idx[1], "MENGE"] = -abs(int(ekbe.loc[idx[1], "MENGE"]))
    # 3. receipt for a PO that does not exist (orphan)
    ekbe.loc[idx[2], "EBELN"] = "4599999999"
    return ekbe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--inject-dq", action="store_true",
                        help="add deliberate data-quality errors")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    suppliers, profiles, weights = make_suppliers(rng)
    materials = make_materials(rng)
    ekko, ekpo, eket = make_purchase_orders(rng, suppliers, materials, weights)
    ekbe = make_goods_receipts(rng, ekko, eket, profiles)
    if args.inject_dq:
        ekbe = inject_data_quality_errors(rng, ekko, ekbe)

    args.out.mkdir(parents=True, exist_ok=True)
    suppliers.to_csv(args.out / "lfa1.csv", index=False)
    materials.drop(columns="BASE_PRICE").to_csv(args.out / "mara.csv", index=False)
    ekko.to_csv(args.out / "ekko.csv", index=False)
    ekpo.to_csv(args.out / "ekpo.csv", index=False)
    eket.to_csv(args.out / "eket.csv", index=False)
    ekbe.to_csv(args.out / "ekbe.csv", index=False)

    truth = {
        "seed": args.seed,
        "as_of": AS_OF.isoformat(),
        "chronically_late_suppliers": [s for s, p in profiles.items() if p == PROFILE_LATE],
        "partial_shipper": next(s for s, p in profiles.items() if p == PROFILE_PARTIAL),
        "seasonal_slow_months": [8, 12],
        "data_quality_errors_injected": args.inject_dq,
    }
    (args.out / "planted_patterns.json").write_text(json.dumps(truth, indent=2))

    print(f"Wrote to {args.out}/: "
          f"{len(suppliers)} suppliers, {len(materials)} materials, {len(ekko)} POs, "
          f"{len(ekpo)} PO items, {len(eket)} schedule lines, {len(ekbe)} goods receipts")


if __name__ == "__main__":
    main()
