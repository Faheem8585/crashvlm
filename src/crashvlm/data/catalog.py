"""Build the test catalog: one row per NCAP test with canonical videos and report metadata."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from crashvlm.data.nhtsa import NHTSAClient
from crashvlm.data.views import select_views

# The NHTSA API stores "not measured" as 0 / "0" for these fields.
ZERO_IS_MISSING = {
    "vehicleTestWeight",
    "vehicleLength",
    "vehicleWidth",
    "vaxCrushDistance",
    "headInjuryCriterion",
    "millisecCliptDoraxRegionPeakAcceleration",
    "leftFemurPeakLoadMeasure",
    "rightFemurPeakLoadMeasure",
    "chestSeverityIndex",
    "shoulderBeltPeakLoadMeasurement",
    "lapBeltPeakLoadMeasurement",
}

NON_TARGET_MAKES = {"NHTSA", "BARRIER"}

SEATS = {
    "LEFT FRONT SEAT": "driver",
    "RIGHT FRONT SEAT": "passenger",
    "LEFT REAR SEAT": "rear_left",
}


def to_float(value: Any, field: str | None = None) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    if field in ZERO_IS_MISSING and x == 0:
        return math.nan
    return x


def test_year(test_date: str | None) -> int | None:
    # Dates come as "2019-00-12" (month often 00), so only the year is reliable.
    try:
        return int(str(test_date)[:4])
    except ValueError:
        return None


def target_vehicle(vehicles: list[dict], occupants: list[dict]) -> dict | None:
    """The vehicle being rated: not the moving barrier, preferably the one carrying dummies."""
    cands = [
        v
        for v in vehicles
        if (v.get("vehicleMake") or "").upper() not in NON_TARGET_MAKES
        and "IMPACTOR" not in (v.get("vehicleModel") or "").upper()
    ]
    if not cands:
        return None
    with_occ = {o.get("vehicleNo") for o in occupants}
    cands.sort(key=lambda v: (v.get("vehicleNo") not in with_occ, v.get("vehicleNo")))
    return cands[0]


def filter_tests(tests: list[dict], test_types: list[str], configs: list[str], min_year: int):
    types = {t.upper() for t in test_types}
    out = []
    for t in tests:
        year = test_year(t.get("testDate"))
        if (
            (t.get("testType") or "").upper() in types
            and t.get("testConfigurationKey") in configs
            and year is not None
            and year >= min_year
        ):
            out.append(t)
    return out


def build_rows(test: dict, media: dict, vehicles: list[dict], occupants: list[dict], max_mb: float):
    """Return (test_row, occupant_rows) for one test."""
    cfg = test["testConfigurationKey"]
    videos = media.get("videos") or []
    views = select_views(cfg, videos, max_bytes=max_mb * 1e6)
    veh = target_vehicle(vehicles, occupants) or {}
    row: dict[str, Any] = {
        "test_no": int(test["testNo"]),
        "test_type": test.get("testType"),
        "config_key": cfg,
        "config": test.get("testConfiguration"),
        "year": test_year(test.get("testDate")),
        "performer": test.get("testPerformer"),
        "closing_speed_kmh": to_float(test.get("closingSpeed")),
        "impact_angle_deg": to_float(test.get("impactAngle")),
        "vehicle_no": veh.get("vehicleNo"),
        "make": veh.get("vehicleMake"),
        "model": veh.get("vehicleModel"),
        "model_year": veh.get("modelYear"),
        "test_weight_kg": to_float(veh.get("vehicleTestWeight"), "vehicleTestWeight"),
        "length_mm": to_float(veh.get("vehicleLength"), "vehicleLength"),
        "width_mm": to_float(veh.get("vehicleWidth"), "vehicleWidth"),
        "crush_mm": to_float(veh.get("vaxCrushDistance"), "vaxCrushDistance"),
        "n_videos": len(videos),
    }
    for role in ("exterior", "occupant"):
        v = views.get(role) or {}
        row[f"{role}_name"] = v.get("originalFileName")
        row[f"{role}_url"] = v.get("media")
        row[f"{role}_bytes"] = to_float(v.get("fileSize"))

    occ_rows = []
    for o in occupants:
        if veh and o.get("vehicleNo") != veh.get("vehicleNo"):
            continue
        seat = SEATS.get((o.get("occupantLocation") or "").upper())
        occ = {
            "test_no": row["test_no"],
            "seat": seat or o.get("occupantLocation"),
            "dummy": o.get("occupantType"),
            "percentile": o.get("dummySizePercentile"),
            "hic": to_float(o.get("headInjuryCriterion"), "headInjuryCriterion"),
            "hic_t1_ms": to_float(o.get("lowerBoundaryofHICTimeInterval")),
            "hic_t2_ms": to_float(o.get("upperBoundaryofHICTimeInterval")),
            "chest_3ms_g": to_float(
                o.get("millisecCliptDoraxRegionPeakAcceleration"),
                "millisecCliptDoraxRegionPeakAcceleration",
            ),
            "femur_left_n": to_float(o.get("leftFemurPeakLoadMeasure"), "leftFemurPeakLoadMeasure"),
            "femur_right_n": to_float(
                o.get("rightFemurPeakLoadMeasure"), "rightFemurPeakLoadMeasure"
            ),
            "shoulder_belt_n": to_float(
                o.get("shoulderBeltPeakLoadMeasurement"), "shoulderBeltPeakLoadMeasurement"
            ),
        }
        occ_rows.append(occ)
        if seat in ("driver", "passenger"):
            for k in ("hic", "chest_3ms_g", "femur_left_n", "femur_right_n"):
                row[f"{seat}_{k}"] = occ[k]
            row[f"{seat}_dummy"] = occ["dummy"]
    return row, occ_rows


def build_catalog(client: NHTSAClient, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = filter_tests(
        client.all_tests(cfg.catalog.workers),
        list(cfg.catalog.test_types),
        list(cfg.catalog.configs),
        cfg.catalog.min_year,
    )

    def fetch(t):
        n = t["testNo"]
        return build_rows(
            t, client.multimedia(n), client.vehicles(n), client.occupants(n), cfg.clips.max_file_mb
        )

    results = client.map(fetch, tests, workers=cfg.catalog.workers)
    rows = [r for r, _ in results]
    occ = [o for _, os_ in results for o in os_]
    cat = pd.DataFrame(rows).sort_values("test_no").reset_index(drop=True)
    cat["has_both_views"] = cat["exterior_url"].notna() & cat["occupant_url"].notna()
    return cat, pd.DataFrame(occ)
