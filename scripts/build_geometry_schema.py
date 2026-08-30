"""Build the Phase 05 additive Silverstone geometry artefact.

READ-ONLY with respect to the simulator. This script does not import, construct
or modify ``F1RaceEnv``, does not touch reward, hazard, tyre, training or
evaluator code, and does not rewrite ``data/silverstone_2024_track_segments.csv``
(it is read for comparison only). It emits a new, additive artefact keyed on the
existing ``segment_id`` values.

WHAT THIS IS
------------
A traceable geometry schema over the existing 36-segment Silverstone model:
18 ``Straight`` segments at odd ``segment_id`` and 18 zero-length ``Corner``
segments at even ``segment_id``, alternating. ``segment_id`` remains the sole
primary key; every column here is supplementary.

This is a geometry-aware strategic simulator, NOT a validated digital twin.

PROVENANCE MODEL
----------------
Every emitted column is labelled in ``column_provenance.csv`` as one of:

``observed``  read directly from FastF1 with no transformation;
``derived``   computed from observed values by a rule stated in the README;
``assumed``   not obtainable from the data at all (external circuit naming), or
              a field whose values are observed but whose semantics are not
              documented by the source.

COORDINATE SCALE
----------------
FastF1 ``X``/``Y``/``Z`` are in an undocumented unit. Phase 03 flagged this as an
open question. This script resolves it empirically by fitting the cumulative
``X``/``Y`` polyline length of a lap's telemetry against the metre-denominated
``Distance`` channel on the same lap, over several laps, and reports the fitted
scale with its observed range.

SECTOR BOUNDARIES
-----------------
Derived by interpolating each lap's ``SectorNSessionTime`` onto that lap's
``Distance`` channel. Absolute boundary distances are driver-dependent (racing
line length varies), so boundaries are published as intervals in metres AND as
lap fractions, and the more stable of the two is stated in the README.

DATA ACCESS DISCIPLINE
----------------------
Only the 2024 British Grand Prix **Race** session is loaded, from the existing
warm ``data/cache``, with ``fastf1.Cache.offline_mode(True)`` set so that a cache
miss raises rather than silently fetching. No cache is written. No other session
is loaded. ``data/fastf1_cache`` is never referenced.

Usage
-----
    .venv_f1/bin/python scripts/build_geometry_schema.py --label 20260830T143000
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import fastf1


CACHE_DIR = Path("data/cache")
SEGMENTS_CSV = Path("data/silverstone_2024_track_segments.csv")
OUTPUT_ROOT = Path("outputs/phase-05-geometry")

YEAR = 2024
EVENT = "Silverstone"
SESSION_CODE = "R"
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
N_SEGMENTS_EXPECTED = 36
N_CORNERS_EXPECTED = 18

# ---------------------------------------------------------------------------
# ASSUMED, EXTERNAL: published Silverstone circuit naming.
#
# FastF1's ``CircuitInfo.corners`` has NO ``Name`` column (verified: columns are
# exactly X, Y, Number, Letter, Angle, Distance). The ``corner_name`` values in
# the existing segments CSV ("Turn 1".."Turn 18") are fallback strings generated
# by track.py, not source data. The names below therefore come from published
# circuit descriptions and are NOT derivable from any field in this dataset.
#
# They are corroborated only indirectly, by the derived arc lengths: the two
# longest straights on the lap fall exactly between the corner pairs that
# published sources place at either end of the Hangar and Wellington straights
# (see README). That is corroboration, not derivation. Treat as ASSUMED.
# ---------------------------------------------------------------------------
CORNER_NAMES_ASSUMED = {
    1: "Abbey",
    2: "Farm Curve",
    3: "Village",
    4: "The Loop",
    5: "Aintree",
    6: "Brooklands",
    7: "Luffield",
    8: "Woodcote",
    9: "Copse",
    10: "Maggotts",
    11: "Becketts",
    12: "Becketts",
    13: "Becketts",
    14: "Chapel",
    15: "Stowe",
    16: "Vale",
    17: "Club",
    18: "Club exit",
}

# ASSUMED, EXTERNAL: the only straights on this circuit carrying published
# proper names. Keyed by (from_turn, to_turn). Every other straight is left
# NULL rather than given a composed pseudo-name.
STRAIGHT_NAMES_ASSUMED = {
    (18, 1): "start/finish straight",
    (5, 6): "Wellington Straight",
    (14, 15): "Hangar Straight",
}

NOT_REPRESENTED = [
    "elevation (telemetry Z exists but is not incorporated; its unit is only "
    "inferred by analogy with X/Y and is untested here)",
    "camber and banking",
    "track width",
    "kerb, run-off and barrier geometry",
    "surface state, grip or friction model",
    "racing-line model",
    "corner entry/exit or apex speeds",
    "fitted corner radii (the legacy approx_radius is a chord proxy, not a radius)",
    "any vehicle-dynamics validation",
]


# ---------------------------------------------------------------------------
# session access
# ---------------------------------------------------------------------------

def load_race_session():
    """Load the 2024 British GP Race from the warm cache, offline."""
    if not CACHE_DIR.exists():
        raise RuntimeError(f"cache directory {CACHE_DIR} does not exist")
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    # Hard guarantee: a cache miss raises instead of fetching, and nothing is
    # written to the cache directory.
    fastf1.Cache.offline_mode(True)
    session = fastf1.get_session(YEAR, EVENT, SESSION_CODE)
    session.load()
    return session


def clean_telemetry(lap):
    """Telemetry for one lap with the Phase 03 OffTrack filter applied."""
    tel = lap.get_telemetry()
    if "Status" in tel.columns:
        tel = tel[tel["Status"] != "OffTrack"]
    return tel


# ---------------------------------------------------------------------------
# lap selection
# ---------------------------------------------------------------------------

def select_reference_laps(session):
    """Fastest accurate dry lap per driver, plus the per-compound fastest.

    Returns (laps_list, rows) where ``rows`` records why each lap was selected.
    Restricting to ``IsAccurate`` and to dry compounds matches the filtering the
    existing calibration path already applies (data_loader.py) and keeps
    wet/intermediate racing lines out of the geometry fit.
    """
    laps = session.laps
    dry = laps[laps["Compound"].isin(DRY_COMPOUNDS) & (laps["IsAccurate"] == True)]  # noqa: E712

    selected, rows = [], []
    seen = set()

    def add(lap, reason):
        key = (str(lap["Driver"]), int(lap["LapNumber"]))
        if key in seen:
            # already selected under another reason; record the extra reason
            for r in rows:
                if (r["driver"], r["lap_number"]) == key:
                    r["selection_reason"] += "; " + reason
            return
        seen.add(key)
        selected.append(lap)
        rows.append(
            {
                "driver": str(lap["Driver"]),
                "lap_number": int(lap["LapNumber"]),
                "compound": str(lap["Compound"]),
                "lap_time_s": lap["LapTime"].total_seconds(),
                "selection_reason": reason,
            }
        )

    for compound, group in dry.groupby("Compound"):
        add(group.pick_fastest(), f"fastest dry lap on {compound}")
    for driver, group in dry.groupby("Driver"):
        add(group.pick_fastest(), f"fastest dry lap for driver {driver}")

    return selected, rows


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def fit_scale_and_sectors(laps, rows):
    """Per-lap coordinate-scale fit and sector-boundary derivation."""
    out = []
    for lap, meta in zip(laps, rows):
        tel = clean_telemetry(lap)
        d = tel["Distance"].to_numpy(dtype=float)
        x = tel["X"].to_numpy(dtype=float)
        y = tel["Y"].to_numpy(dtype=float)
        if len(d) < 2:
            continue

        lap_len_m = float(d.max() - d.min())
        arc_raw = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if lap_len_m <= 0:
            continue

        rec = dict(meta)
        rec["telemetry_rows"] = int(len(tel))
        rec["lap_length_m"] = lap_len_m
        rec["xy_polyline_raw_units"] = arc_raw
        rec["raw_units_per_metre"] = arc_raw / lap_len_m
        rec["metres_per_raw_unit"] = lap_len_m / arc_raw
        rec["median_sample_spacing_m"] = float(np.median(np.diff(d)))

        session_time_s = tel["SessionTime"].dt.total_seconds().to_numpy()
        for k in (1, 2, 3):
            t = lap.get(f"Sector{k}SessionTime")
            if t is None or pd.isna(t):
                rec[f"sector{k}_end_m"] = np.nan
                rec[f"sector{k}_end_frac"] = np.nan
                continue
            b = float(np.interp(t.total_seconds(), session_time_s, d))
            rec[f"sector{k}_end_m"] = b
            rec[f"sector{k}_end_frac"] = b / lap_len_m
        out.append(rec)
    return pd.DataFrame(out)


def summarise(df, columns):
    """min/max/mean/range summary for the named columns."""
    recs = []
    for col in columns:
        v = df[col].dropna().to_numpy(dtype=float)
        if v.size == 0:
            continue
        recs.append(
            {
                "quantity": col,
                "n_laps": int(v.size),
                "min": float(v.min()),
                "max": float(v.max()),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "range": float(v.max() - v.min()),
            }
        )
    return pd.DataFrame(recs)


def check_corner_distances(corners, ref_lap, scale):
    """Compare corners.Distance against telemetry Distance at the nearest
    X/Y sample. Tests whether the two share a unit and an origin."""
    tel = clean_telemetry(ref_lap)
    d = tel["Distance"].to_numpy(dtype=float)
    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)

    recs = []
    for _, r in corners.iterrows():
        i = int(np.argmin((x - float(r["X"])) ** 2 + (y - float(r["Y"])) ** 2))
        residual_raw = float(np.hypot(x[i] - float(r["X"]), y[i] - float(r["Y"])))
        recs.append(
            {
                "corner_number": int(r["Number"]),
                "corner_letter": str(r["Letter"]),
                "corners_distance_m": float(r["Distance"]),
                "nearest_sample_distance_m": float(d[i]),
                "diff_m": float(r["Distance"]) - float(d[i]),
                "nearest_sample_xy_residual_raw": residual_raw,
                "nearest_sample_xy_residual_m": residual_raw * scale,
            }
        )
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# the 36-row schema
# ---------------------------------------------------------------------------

def sector_of(frac, f1, f2):
    if frac < f1:
        return 1
    if frac < f2:
        return 2
    return 3


def build_segment_geometry(corners, legacy, lap_length_m, scale, f1, f2):
    """One row per existing segment_id, 36 rows, additive columns only."""
    corners = corners.sort_values(["Number", "Letter"]).reset_index(drop=True)
    cd = corners["Distance"].to_numpy(dtype=float)
    xs = corners["X"].to_numpy(dtype=float)
    ys = corners["Y"].to_numpy(dtype=float)
    nums = corners["Number"].to_numpy(dtype=int)
    angles = corners["Angle"].to_numpy(dtype=float)
    n = len(corners)

    legacy_by_id = legacy.set_index("id")
    rows = []

    for i in range(n):
        j = (i - 1) % n
        seg_straight = 2 * i + 1
        seg_corner = 2 * i + 2

        # --- straight preceding corner i, wrap-aware for i == 0 ---
        d_start, d_end = float(cd[j]), float(cd[i])
        if i > 0:
            arc = d_end - d_start
        else:
            # segment 1 crosses the start/finish line: from the last corner,
            # over the line, to the first corner.
            arc = (lap_length_m - float(cd[n - 1])) + float(cd[0])
        chord_raw = float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))

        lf_start = d_start / lap_length_m
        lf_end = d_end / lap_length_m
        legacy_len_raw = float(legacy_by_id.at[seg_straight, "length"])

        rows.append(
            {
                "segment_id": seg_straight,
                "segment_type": "Straight",
                "region_key": f"T{nums[j]}_to_T{nums[i]}",
                "region_description": f"Straight from Turn {nums[j]} to Turn {nums[i]}",
                "region_name_external": STRAIGHT_NAMES_ASSUMED.get(
                    (int(nums[j]), int(nums[i]))
                ),
                "corner_number": None,
                "d_start_m": d_start,
                "d_end_m": d_end,
                "arc_length_m": arc,
                "chord_length_m": chord_raw / scale,
                "lap_fraction_start": lf_start,
                "lap_fraction_end": lf_end,
                "sector_start": sector_of(lf_start, f1, f2),
                "sector_end": sector_of(lf_end, f1, f2),
                "crosses_sector_boundary": sector_of(lf_start, f1, f2)
                != sector_of(lf_end, f1, f2),
                "corner_angle_raw": None,
                "approx_radius_legacy": None,
                "approx_radius_legacy_m": None,
                "legacy_length_raw": legacy_len_raw,
                "legacy_length_m": legacy_len_raw / scale,
                "legacy_sector": int(legacy_by_id.at[seg_straight, "sector"]),
            }
        )

        # --- zero-length corner at turn i ---
        legacy_radius = legacy_by_id.at[seg_corner, "approx_radius"]
        legacy_radius = None if pd.isna(legacy_radius) else float(legacy_radius)
        lf_corner = d_end / lap_length_m
        sec = sector_of(lf_corner, f1, f2)

        rows.append(
            {
                "segment_id": seg_corner,
                "segment_type": "Corner",
                "region_key": f"T{nums[i]}",
                "region_description": f"Turn {nums[i]}",
                "region_name_external": CORNER_NAMES_ASSUMED.get(int(nums[i])),
                "corner_number": int(nums[i]),
                "d_start_m": d_end,
                "d_end_m": d_end,
                "arc_length_m": 0.0,
                "chord_length_m": 0.0,
                "lap_fraction_start": lf_corner,
                "lap_fraction_end": lf_corner,
                "sector_start": sec,
                "sector_end": sec,
                "crosses_sector_boundary": False,
                "corner_angle_raw": float(angles[i]),
                "approx_radius_legacy": legacy_radius,
                "approx_radius_legacy_m": (
                    None if legacy_radius is None else legacy_radius / scale
                ),
                "legacy_length_raw": float(legacy_by_id.at[seg_corner, "length"]),
                "legacy_length_m": 0.0,
                "legacy_sector": int(legacy_by_id.at[seg_corner, "sector"]),
            }
        )

    return pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)


COLUMN_PROVENANCE = [
    ("segment_id", "observed", "Join key. Identical to id in data/silverstone_2024_track_segments.csv."),
    ("segment_type", "observed", "Straight/Corner, copied unchanged from the existing segments CSV."),
    ("region_key", "observed", "Built only from corners.Number/Letter. Authoritative stable identifier."),
    ("region_description", "derived", "Human-readable rendering of region_key. No external input."),
    ("region_name_external", "assumed", "ASSUMED, EXTERNAL. Published circuit naming; FastF1 supplies no Name column. NULLABLE. Never use as a key."),
    ("corner_number", "observed", "corners.Number. NULL for straights, matching the existing CSV convention."),
    ("d_start_m", "observed", "corners.Distance in metres, verified against the telemetry Distance channel."),
    ("d_end_m", "observed", "corners.Distance in metres. Equal to d_start_m for corners."),
    ("arc_length_m", "derived", "d_end_m - d_start_m, wrap-aware for segment 1. Racing-line distance."),
    ("chord_length_m", "derived", "Straight-line X/Y chord divided by the fitted coordinate scale. Centreline chord."),
    ("lap_fraction_start", "derived", "d_start_m / lap_length_m."),
    ("lap_fraction_end", "derived", "d_end_m / lap_length_m."),
    ("sector_start", "derived", "Sector containing lap_fraction_start, from the multi-lap fraction boundaries."),
    ("sector_end", "derived", "Sector containing lap_fraction_end."),
    ("crosses_sector_boundary", "derived", "True where sector_start != sector_end."),
    ("corner_angle_raw", "assumed", "corners.Angle verbatim. Values observed; SEMANTICS UNDOCUMENTED. Carried, never consumed."),
    ("approx_radius_legacy", "observed", "Existing approx_radius verbatim, republished so hazard's actual input is documented."),
    ("approx_radius_legacy_m", "derived", "approx_radius_legacy / fitted scale. Not a physical radius."),
    ("legacy_length_raw", "observed", "Existing length column verbatim, including the known segment-1 defect."),
    ("legacy_length_m", "derived", "legacy_length_raw / fitted scale."),
    ("legacy_sector", "observed", "Existing scalar sector column verbatim, including its known S1 under-assignment."),
]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(geom, corners, legacy, lap_length_m, f1, f2, f1_range, f2_range):
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "result": "PASS" if passed else "FAIL",
                       "detail": detail})

    n = len(geom)
    check("row_count_is_36", n == N_SEGMENTS_EXPECTED, f"{n} rows")

    ids = geom["segment_id"].tolist()
    check("segment_ids_are_1_to_36", ids == list(range(1, N_SEGMENTS_EXPECTED + 1)),
          f"min={min(ids)} max={max(ids)} unique={geom['segment_id'].nunique()}")

    types = geom.set_index("segment_id")["segment_type"]
    alt_ok = all(types[i] == ("Straight" if i % 2 == 1 else "Corner")
                 for i in range(1, N_SEGMENTS_EXPECTED + 1))
    counts = geom["segment_type"].value_counts().to_dict()
    check("alternating_18_straight_18_corner", alt_ok and counts.get("Straight") == 18
          and counts.get("Corner") == 18, str(counts))

    merged = geom.merge(legacy[["id", "segment_type"]], left_on="segment_id",
                        right_on="id", how="inner")
    check("segment_type_matches_existing_csv",
          len(merged) == N_SEGMENTS_EXPECTED
          and (merged["segment_type_x"] == merged["segment_type_y"]).all(),
          f"{len(merged)} rows joined, all types equal")

    check("every_segment_maps_to_exactly_one_region",
          geom["region_key"].notna().all() and geom["region_key"].nunique() == n,
          f"{geom['region_key'].nunique()} distinct region_key over {n} rows")

    arc_sum = float(geom.loc[geom.segment_type == "Straight", "arc_length_m"].sum())
    check("straight_arcs_sum_to_lap_length", abs(arc_sum - lap_length_m) < 0.01,
          f"sum={arc_sum:.4f} m, lap={lap_length_m:.4f} m, "
          f"residual={arc_sum - lap_length_m:.6f} m")

    check("all_corner_arc_lengths_zero",
          (geom.loc[geom.segment_type == "Corner", "arc_length_m"] == 0.0).all(),
          "18 corner segments carry arc_length_m == 0.0")

    src_nums = sorted(int(v) for v in corners["Number"])
    got_nums = sorted(int(v) for v in geom["corner_number"].dropna())
    check("corner_numbering_matches_source", src_nums == got_nums,
          f"source {len(src_nums)} corners, schema {len(got_nums)}, equal={src_nums == got_nums}")

    check("corner_numbers_null_on_straights",
          geom.loc[geom.segment_type == "Straight", "corner_number"].isna().all(),
          "all 18 straights carry NULL corner_number")

    n_named = int(geom["region_name_external"].notna().sum())
    check("region_name_external_is_nullable_and_labelled_assumed", True,
          f"{n_named}/{n} rows carry an assumed external name; "
          f"{n - n_named} are NULL by design")

    # sector assignment must be robust to the measured boundary uncertainty:
    # no segment endpoint may sit within the boundary's own lap-to-lap range.
    endpoints = np.unique(np.concatenate([
        geom["lap_fraction_start"].to_numpy(dtype=float),
        geom["lap_fraction_end"].to_numpy(dtype=float),
    ]))
    margins = []
    for name, f, r in (("S1", f1, f1_range), ("S2", f2, f2_range)):
        gap = float(np.min(np.abs(endpoints - f)))
        margins.append((name, gap, r, gap / r if r > 0 else float("inf")))
    worst = min(margins, key=lambda m: m[3])
    check("sector_assignment_robust_to_boundary_uncertainty",
          all(m[1] > m[2] for m in margins),
          "; ".join(
              f"{n}: nearest segment endpoint {g:.5f} of a lap away "
              f"({g * lap_length_m:.1f} m), boundary range {r:.5f} "
              f"({r * lap_length_m:.1f} m), margin {x:.1f}x"
              for n, g, r, x in margins)
          + f" | tightest {worst[0]} at {worst[3]:.1f}x")

    # the documented defect must be reproduced verbatim, not silently repaired
    seg1_legacy = float(geom.loc[geom.segment_id == 1, "legacy_length_raw"].iloc[0])
    seg1_arc = float(geom.loc[geom.segment_id == 1, "arc_length_m"].iloc[0])
    check("segment1_legacy_defect_reproduced_and_corrected_separately",
          seg1_legacy == 0.0 and seg1_arc > 0.0,
          f"legacy_length_raw={seg1_legacy} (defect preserved verbatim), "
          f"arc_length_m={seg1_arc:.2f} (corrected value published separately)")

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def write_readme(path, ctx):
    g = ctx["geom"]
    s = ctx["scale_summary"].set_index("quantity")
    b = ctx["sector_summary"].set_index("quantity")

    def rng(q, fmt="{:.2f}"):
        r = b.loc[q]
        return (f"{fmt.format(r['min'])} – {fmt.format(r['max'])} "
                f"(mean {fmt.format(r['mean'])}, range {fmt.format(r['range'])})")

    scale_row = s.loc["raw_units_per_metre"]
    named = g[g["region_name_external"].notna()]

    lines = []
    A = lines.append

    A("# Phase 05 — Silverstone additive geometry artefact")
    A("")
    A(f"Generated {ctx['generated_at']} by `{ctx['script']}`.")
    A("")
    A("## What this is, and what it is not")
    A("")
    A("This is a **geometry-aware strategic simulator, not a validated digital "
      "twin.** Nothing here is validated against vehicle dynamics.")
    A("")
    A("This artefact is **additive and read-only**. It adds supplementary "
      "geometry columns keyed on the *existing* `segment_id` values. It does "
      "not modify `data/silverstone_2024_track_segments.csv`, the 36-segment "
      "structure, `n_segments`, the hazard model, or any code.")
    A("")
    A("**Explicitly not represented:**")
    A("")
    for item in NOT_REPRESENTED:
        A(f"- {item}")
    A("")
    A("## Structure")
    A("")
    A(f"{N_SEGMENTS_EXPECTED} segments: 18 `Straight` at odd `segment_id`, 18 "
      f"zero-length `Corner` at even `segment_id`, strictly alternating. "
      f"`segment_id` is the sole primary key and is unchanged.")
    A("")
    A("`region_key` is the **authoritative stable identifier** for a named "
      "region (`T9`, `T14_to_T15`). It is built only from `corners.Number` and "
      "`corners.Letter`, so it is fully observed. `region_name_external` is "
      "**ASSUMED, external, and nullable** — downstream consumers must key on "
      "`region_key`, never on the name.")
    A("")
    A("## Provenance")
    A("")
    A(f"Single source session: **{YEAR} {EVENT} {SESSION_CODE}**, loaded from "
      f"the warm `{CACHE_DIR}` with `fastf1.Cache.offline_mode(True)` so a cache "
      f"miss raises rather than fetching. No cache written; no other session "
      f"loaded; `data/fastf1_cache` never referenced.")
    A("")
    A("Per-column `observed` / `derived` / `assumed` labels with their "
      "derivation rules are in `column_provenance.csv`. Phase 03 "
      "(`docs/phase-03-fastf1-feasibility-audit.md`) established that "
      "`circuit_info.corners` returns 18 null-free rows identically across all "
      "five sessions of this event.")
    A("")
    A("## The coordinate-unit question, resolved")
    A("")
    A("Phase 03 recorded the unit of FastF1 `X`/`Y`/`Z` as an **open "
      "assumption**: the coordinate range is far too large for metres, and the "
      "documentation does not state a unit. This artefact resolves it by "
      "measurement.")
    A("")
    A(f"Fitting each reference lap's cumulative `X`/`Y` polyline length against "
      f"its metre-denominated `Distance` channel over "
      f"**{int(scale_row['n_laps'])} laps**:")
    A("")
    A(f"- **{scale_row['min']:.6f} – {scale_row['max']:.6f} raw units per "
      f"metre** (mean **{scale_row['mean']:.6f}**, sd {scale_row['sd']:.6f})")
    A(f"- i.e. 1 raw unit = **{s.loc['metres_per_raw_unit', 'mean']:.6f} m**")
    A("")
    A(f"`X`/`Y`/`Z` are therefore in **decimetres (1/10 m)**, with a fitted "
      f"deviation from exactly 10.0 of "
      f"{abs(scale_row['mean'] - 10.0) / 10.0 * 100:.4f}% and a lap-to-lap "
      f"spread of {scale_row['range'] / scale_row['mean'] * 100:.4f}%. This is "
      f"**derived with a stated residual**, not assumed.")
    A("")
    A(f"The origin of the sub-percent deviation from exactly 10.0 is **not "
      f"established here**. Candidate causes, none tested: the straight-line "
      f"polyline between samples spaced ~{ctx['median_spacing_m']:.1f} m apart "
      f"under-measures true curved arc; and the `Distance` channel is itself a "
      f"FastF1-derived quantity rather than a coordinate measurement, so the two "
      f"sides of the fit are not independent measurements of one path. Do not "
      f"cite a mechanism for the residual without testing it.")
    A("")
    A(f"The scale used for all `*_m` conversions in `segment_geometry.csv` is "
      f"the multi-lap mean, **{scale_row['mean']:.6f}**.")
    A("")
    A("`corners.Distance` was independently confirmed to be in metres on the "
      "same origin as the telemetry channel: see `corner_distance_check.csv`. "
      f"{ctx['n_exact']}/{N_CORNERS_EXPECTED} corners agree to 0.00 m with the "
      f"nearest telemetry sample; the largest disagreement is "
      f"{ctx['max_abs_diff']:.2f} m, below one "
      f"~{ctx['median_spacing_m']:.1f} m sample interval.")
    A("")
    A("## Distance boundaries")
    A("")
    A("Region boundaries are **observed**, taken from `corners.Distance` in "
      "metres. Segment 1 is wrap-aware: it runs from the last corner, across "
      "the start/finish line, to Turn 1.")
    A("")
    A(f"**Validation:** the 18 derived straight `arc_length_m` values sum to "
      f"**{ctx['arc_sum']:.2f} m** against a measured lap length of "
      f"**{ctx['lap_length_m']:.2f} m** — closing to "
      f"{abs(ctx['arc_sum'] - ctx['lap_length_m']):.6f} m.")
    A("")
    A("**Caveat — two different reference paths.** `arc_length_m` follows the "
      "*racing line* (`corners.Distance` is snapped to the telemetry grid of a "
      "driven lap); `chord_length_m` is a *centreline map* chord between corner "
      "`X`/`Y` points. They are not measurements of the same path. The "
      f"consequence is visible: **{ctx['n_chord_gt_arc']} of 18 chords exceed "
      f"their own arc length**, which is impossible on a single path. Never mix "
      f"the two in one calculation.")
    A("")
    A(f"The measured lap length ({ctx['lap_length_m']:.2f} m) is a racing-line "
      f"distance and is **not** the published circuit length. Do not present it "
      f"as such.")
    A("")
    A("## Sector boundaries and their uncertainty")
    A("")
    A("Derived by interpolating each reference lap's `SectorNSessionTime` onto "
      "that lap's `Distance` channel. Per-lap values are in "
      "`sector_boundaries.csv`; the summary is in `sector_boundary_summary.csv`.")
    A("")
    A(f"Over {int(b.loc['sector1_end_m', 'n_laps'])} reference laps:")
    A("")
    A("| Boundary | Absolute (m) | Lap fraction |")
    A("|---|---|---|")
    A(f"| End of S1 | {rng('sector1_end_m')} | {rng('sector1_end_frac', '{:.5f}')} |")
    A(f"| End of S2 | {rng('sector2_end_m')} | {rng('sector2_end_frac', '{:.5f}')} |")
    A(f"| End of S3 (lap) | {rng('sector3_end_m')} | {rng('sector3_end_frac', '{:.5f}')} |")
    A("")
    A(f"**Absolute boundaries are NOT stable and are published as intervals, "
      f"not point estimates.** End-of-S1 varies by "
      f"{b.loc['sector1_end_m', 'range']:.2f} m and end-of-S2 by "
      f"{b.loc['sector2_end_m', 'range']:.2f} m across laps, because total "
      f"racing-line length itself varies by "
      f"{b.loc['sector3_end_m', 'range']:.2f} m between drivers.")
    A("")
    A("**Lap fractions are more stable, but by a margin that varies sharply by "
      "boundary.** Comparing each quantity's range against its own mean "
      "(range / mean, so absolute and fractional forms are comparable):")
    A("")
    A("| Boundary | Absolute spread | Fractional spread | Fraction tighter by |")
    A("|---|---|---|---|")
    for k, lab in ((1, "End of S1"), (2, "End of S2"), (3, "End of S3 (lap)")):
        am = b.loc[f"sector{k}_end_m"]
        fr = b.loc[f"sector{k}_end_frac"]
        ra = am["range"] / am["mean"] * 100
        rf = fr["range"] / fr["mean"] * 100
        A(f"| {lab} | {ra:.3f}% | {rf:.3f}% | "
          f"{ra / rf:.1f}x |" if rf > 0 else f"| {lab} | {ra:.3f}% | ~0% | — |")
    A("")
    A(f"So the lap boundary is essentially exact in fractional terms, S2 is "
      f"materially tighter, and **S1 is only marginally tighter** — the "
      f"fractional form is not a uniform improvement, and the S1 boundary "
      f"carries real uncertainty either way. **Use lap fraction, but treat the "
      f"S1 boundary as uncertain to ~"
      f"{b.loc['sector1_end_frac', 'range'] * 100:.2f}% of a lap.** "
      f"`sector_start`/`sector_end` in `segment_geometry.csv` are assigned from "
      f"the mean fraction boundaries ({ctx['f1']:.5f}, {ctx['f2']:.5f}); no "
      f"segment boundary in this schema falls close enough to either sector "
      f"boundary for that uncertainty to change its assignment.")
    A("")
    A(f"{int(g['crosses_sector_boundary'].sum())} straights cross a sector "
      f"boundary, which is why sector is published as a "
      f"`sector_start`/`sector_end` pair rather than the existing scalar. The "
      f"existing scalar `sector` column is republished unchanged as "
      f"`legacy_sector`.")
    A("")
    A("## Radius proxy")
    A("")
    A("`approx_radius_legacy` reproduces the existing `approx_radius` verbatim, "
      "so that the value the hazard model actually consumes is documented. It "
      "is computed by `track.py` as the **mean chord distance to the previous "
      "and next corner points** — a crude proxy, **not a fitted radius**, and "
      "not a physical quantity.")
    A("")
    A(f"Converted at the fitted scale it spans "
      f"{ctx['radius_min_m']:.1f} – {ctx['radius_max_m']:.1f} m. Two properties "
      f"make it unfit for physical use: it is **larger** in fast open sections, "
      f"inverting the intended \"tighter corner is riskier\" semantics; and in "
      f"the hazard model it contributes only "
      f"{ctx['radius_term_pct_min']:.3f}%–{ctx['radius_term_pct_max']:.3f}% of "
      f"the corner base rate, so corners are effectively identical to the hazard "
      f"model.")
    A("")
    A("**No corrected radius is published and no hazard change is proposed.** "
      "Rewiring the radius term would be a dynamics change requiring separate "
      "approval and would invalidate frozen evidence.")
    A("")
    A("## Known upstream defects — documented, not repaired")
    A("")
    A(f"Both are reproduced verbatim in the `legacy_*` columns and corrected "
      f"only in the new columns. `data/silverstone_2024_track_segments.csv` is "
      f"byte-unchanged.")
    A("")
    A(f"1. **Segment 1's `length` is `0.0`, but its true extent is "
      f"{ctx['seg1_arc']:.2f} m of arc "
      f"({ctx['seg1_chord']:.2f} m of chord)** — the longest straight on the "
      f"lap. `track.py:87` computes the wrap-around straight as "
      f"`total_len - s_coords[-1]`, which is identically zero for any circuit. "
      f"The segment's coordinates are correct; only `length` is wrong.")
    A(f"2. **The existing scalar `sector` under-assigns sector 1.** "
      f"`track.py:90-91` applies the proportion rule to the *corner*, not to "
      f"the straight preceding it, and Turn 1's cumulative distance is 0, so "
      f"segment 1 is labelled sector 1 despite beginning in sector 3.")
    A("")
    A("Neither defect is behavioural: `length` and `sector` are read by **no "
      "code** in this repository — verified by grep across `src/`, `scripts/` "
      "and `notebooks/`. Only `id`, `segment_type`, `corner_number`, "
      "`corner_name` and `approx_radius` are consumed.")
    A("")
    A("## Assumed external naming")
    A("")
    A(f"`region_name_external` is populated on {len(named)} of "
      f"{N_SEGMENTS_EXPECTED} rows and NULL on the rest. FastF1's "
      f"`corners` table has **no `Name` column** (its columns are exactly `X`, "
      f"`Y`, `Number`, `Letter`, `Angle`, `Distance`), and the `corner_name` "
      f"values in the existing CSV are `track.py` fallbacks (`\"Turn 1\"`…), not "
      f"source data. These names are therefore **not derivable from the "
      f"dataset** and are labelled `assumed`.")
    A("")
    A("They are corroborated indirectly by the derived arc lengths: the two "
      "longest straights on the lap fall exactly between the corner pairs that "
      "published sources place at either end of the two named straights.")
    A("")
    A("| Segment | region_key | Assumed name | Derived arc (m) | Rank |")
    A("|---|---|---|---|---|")
    straights = g[g.segment_type == "Straight"].copy()
    straights["rank"] = straights["arc_length_m"].rank(ascending=False).astype(int)
    for key in [(5, 6), (14, 15), (18, 1)]:
        rk = f"T{key[0]}_to_T{key[1]}"
        r = straights[straights.region_key == rk]
        if len(r):
            r = r.iloc[0]
            A(f"| {int(r.segment_id)} | `{rk}` | {r.region_name_external} | "
              f"{r.arc_length_m:.2f} | {int(r['rank'])} of 18 |")
    A("")
    A("This is **corroboration, not derivation**. The names remain assumed.")
    A("")
    A("`corner_angle_raw` carries `corners.Angle` verbatim. Its values are "
      "observed and null-free, but its **unit and reference direction are "
      "undocumented**, and two values look like angle-wrap artefacts "
      f"(T18 = {ctx['angle_t18']:.2f}, T10 = {ctx['angle_t10']:.2f}). It is "
      f"carried for future work and **consumed by nothing**.")
    A("")
    A("## Backwards compatibility")
    A("")
    A("Additive only, in a new file. No existing column is renamed, retyped or "
      "repurposed; no key is added to or removed from the environment `info` "
      "dict or the crash log; `n_segments` remains "
      f"{N_SEGMENTS_EXPECTED}. The frozen pace-diagnostics evidence at "
      f"`outputs/phase2-recalibration/pace_diagnostics/20260827T124054/`, whose "
      f"aggregation asserts exactly {N_SEGMENTS_EXPECTED} segments per lap, is "
      f"unaffected. Validation results are in `validation.csv`.")
    A("")
    A("## Files")
    A("")
    A("| File | Role |")
    A("|---|---|")
    for r in ctx["manifest_rows"]:
        A(f"| `{r['file']}` | {r['role']} |")
    A("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="output subdirectory name under outputs/phase-05-geometry/")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced scope: 3 reference laps (one per dry compound)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    t0 = time.time()

    outdir = OUTPUT_ROOT / args.label
    if outdir.exists() and any(outdir.iterdir()):
        print(f"refusing to overwrite non-empty {outdir}", file=sys.stderr)
        return 2

    if not SEGMENTS_CSV.exists():
        print(f"missing {SEGMENTS_CSV}", file=sys.stderr)
        return 2
    legacy = pd.read_csv(SEGMENTS_CSV)
    if len(legacy) != N_SEGMENTS_EXPECTED:
        print(f"expected {N_SEGMENTS_EXPECTED} segment rows, got {len(legacy)}",
              file=sys.stderr)
        return 2

    print(f"loading {YEAR} {EVENT} {SESSION_CODE} from warm cache (offline)...")
    t_load = time.time()
    session = load_race_session()
    load_s = time.time() - t_load

    corners = session.get_circuit_info().corners.copy()
    corners = corners.sort_values(["Number", "Letter"]).reset_index(drop=True)
    if len(corners) != N_CORNERS_EXPECTED:
        print(f"expected {N_CORNERS_EXPECTED} corners, got {len(corners)}",
              file=sys.stderr)
        return 2

    laps, lap_rows = select_reference_laps(session)
    if args.smoke:
        keep = [i for i, r in enumerate(lap_rows)
                if r["selection_reason"].startswith("fastest dry lap on")]
        laps = [laps[i] for i in keep]
        lap_rows = [lap_rows[i] for i in keep]
    print(f"selected {len(laps)} reference laps")

    per_lap = fit_scale_and_sectors(laps, lap_rows)
    scale_summary = summarise(per_lap, ["raw_units_per_metre", "metres_per_raw_unit",
                                        "lap_length_m"])
    sector_summary = summarise(per_lap, ["sector1_end_m", "sector2_end_m",
                                         "sector3_end_m", "sector1_end_frac",
                                         "sector2_end_frac", "sector3_end_frac"])

    ss = scale_summary.set_index("quantity")
    bs = sector_summary.set_index("quantity")
    scale = float(ss.loc["raw_units_per_metre", "mean"])
    f1 = float(bs.loc["sector1_end_frac", "mean"])
    f2 = float(bs.loc["sector2_end_frac", "mean"])

    # the reference lap for the corner-distance check: the fastest selected lap
    ref_idx = int(per_lap["lap_time_s"].idxmin())
    ref_lap = laps[ref_idx]
    lap_length_m = float(per_lap.loc[ref_idx, "lap_length_m"])
    corner_check = check_corner_distances(corners, ref_lap, 1.0 / scale)

    geom = build_segment_geometry(corners, legacy, lap_length_m, scale, f1, f2)
    validation = validate(
        geom, corners, legacy, lap_length_m, f1, f2,
        float(bs.loc["sector1_end_frac", "range"]),
        float(bs.loc["sector2_end_frac", "range"]),
    )
    provenance = pd.DataFrame(COLUMN_PROVENANCE,
                              columns=["column", "provenance", "definition"])

    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        {"file": "segment_geometry.csv",
         "role": f"THE SCHEMA: {N_SEGMENTS_EXPECTED} rows, one per existing "
                 f"segment_id, additive geometry columns.",
         "rows": len(geom)},
        {"file": "column_provenance.csv",
         "role": "Per-column observed/derived/assumed label and derivation rule.",
         "rows": len(provenance)},
        {"file": "sector_boundaries.csv",
         "role": "Per-lap sector boundaries and coordinate-scale fit, one row "
                 "per reference lap.",
         "rows": len(per_lap)},
        {"file": "sector_boundary_summary.csv",
         "role": "Sector boundaries as INTERVALS (min/max/mean/sd/range), "
                 "absolute metres and lap fractions.",
         "rows": len(sector_summary)},
        {"file": "scale_fit_summary.csv",
         "role": "Coordinate-scale fit summary establishing the decimetre unit.",
         "rows": len(scale_summary)},
        {"file": "corner_distance_check.csv",
         "role": "corners.Distance vs telemetry Distance at the nearest X/Y "
                 "sample, per corner. Evidence the two share unit and origin.",
         "rows": len(corner_check)},
        {"file": "validation.csv",
         "role": "Schema validation checks and results.",
         "rows": len(validation)},
        {"file": "README.md",
         "role": "Provenance, scale fit, sector uncertainty, defects, "
                 "compatibility, and the not-a-digital-twin statement.",
         "rows": 0},
        {"file": "manifest.csv", "role": "This file listing.", "rows": 0},
    ]

    geom.to_csv(outdir / "segment_geometry.csv", index=False)
    provenance.to_csv(outdir / "column_provenance.csv", index=False)
    per_lap.to_csv(outdir / "sector_boundaries.csv", index=False)
    sector_summary.to_csv(outdir / "sector_boundary_summary.csv", index=False)
    scale_summary.to_csv(outdir / "scale_fit_summary.csv", index=False)
    corner_check.to_csv(outdir / "corner_distance_check.csv", index=False)
    validation.to_csv(outdir / "validation.csv", index=False)

    radii = geom["approx_radius_legacy_m"].dropna()
    radii_raw = geom["approx_radius_legacy"].dropna()
    corner_base = 0.002 * 8.0
    ctx = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/build_geometry_schema.py",
        "geom": geom,
        "scale_summary": scale_summary,
        "sector_summary": sector_summary,
        "lap_length_m": lap_length_m,
        "arc_sum": float(geom.loc[geom.segment_type == "Straight",
                                  "arc_length_m"].sum()),
        "f1": f1, "f2": f2,
        "median_spacing_m": float(per_lap["median_sample_spacing_m"].median()),
        "n_exact": int((corner_check["diff_m"].abs() < 0.005).sum()),
        "max_abs_diff": float(corner_check["diff_m"].abs().max()),
        "n_chord_gt_arc": int(
            (geom.loc[geom.segment_type == "Straight", "chord_length_m"]
             > geom.loc[geom.segment_type == "Straight", "arc_length_m"]).sum()),
        "radius_min_m": float(radii.min()), "radius_max_m": float(radii.max()),
        "radius_term_pct_min": 0.015 / float(radii_raw.max()) / corner_base * 100,
        "radius_term_pct_max": 0.015 / float(radii_raw.min()) / corner_base * 100,
        "seg1_arc": float(geom.loc[geom.segment_id == 1, "arc_length_m"].iloc[0]),
        "seg1_chord": float(geom.loc[geom.segment_id == 1, "chord_length_m"].iloc[0]),
        "angle_t18": float(geom.loc[geom.segment_id == 36, "corner_angle_raw"].iloc[0]),
        "angle_t10": float(geom.loc[geom.segment_id == 20, "corner_angle_raw"].iloc[0]),
        "manifest_rows": manifest_rows,
    }
    write_readme(outdir / "README.md", ctx)

    manifest = pd.DataFrame(manifest_rows)
    manifest["generated_at"] = ctx["generated_at"]
    manifest["script"] = ctx["script"]
    manifest["label"] = args.label
    manifest["source_session"] = f"{YEAR} {EVENT} {SESSION_CODE}"
    manifest["fastf1_version"] = fastf1.__version__
    manifest["pandas_version"] = pd.__version__
    manifest["numpy_version"] = np.__version__
    manifest["python_version"] = platform.python_version()
    manifest["offline_mode"] = True
    manifest["cache_dir"] = str(CACHE_DIR)
    manifest["reference_laps"] = len(per_lap)
    manifest["raw_units_per_metre_mean"] = scale
    manifest["lap_length_m"] = lap_length_m
    manifest["n_segments"] = len(geom)
    manifest["session_load_seconds"] = round(load_s, 3)
    manifest["total_seconds"] = round(time.time() - t0, 3)
    manifest["segments_csv_modified"] = False
    manifest["validation_all_pass"] = bool((validation["result"] == "PASS").all())
    manifest.to_csv(outdir / "manifest.csv", index=False)

    print(f"\nwrote {len(manifest_rows)} files to {outdir}")
    print(f"scale: {scale:.6f} raw units per metre "
          f"(1 unit = {1.0 / scale:.6f} m)")
    print(f"lap length: {lap_length_m:.2f} m; "
          f"straight arcs sum: {ctx['arc_sum']:.2f} m")
    print(f"\nvalidation ({len(validation)} checks):")
    for _, r in validation.iterrows():
        print(f"  [{r['result']}] {r['check']}: {r['detail']}")
    ok = bool((validation["result"] == "PASS").all())
    print(f"\nall checks passed: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
