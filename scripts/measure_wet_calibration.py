"""Phase 04 wet-session calibration measurement — READ-ONLY.

Implements the pre-registered Phase 04 measurement protocol
(docs/phase-04-wet-session-evidence.md, sections P0-P11). This script
measures; it changes nothing. It does not import, construct or modify
``F1RaceEnv``, and it writes no file outside its own output directory.

Data source (protocol P1, Variant B)
------------------------------------
Primary calibration sample: ``data/silverstone_2024_laps.csv`` — the exact
file ``F1RaceEnv._load_calibration_2024`` reads, produced by
``data_loader.extract_stint_and_lap_data(2024)`` which applies
``laps[laps["IsAccurate"] == True]`` to the 2024 British Grand Prix **Race**
session and keeps 12 columns.

Auxiliary columns are re-read from the *same already-cached Race session*
(``LapStartTime``, ``Deleted``, ``PitInTime``/``PitOutTime``, ``TrackStatus``,
``Position``) plus ``session.weather_data`` (``Rainfall``, ``TrackTemp``,
``AirTemp``). No other session is loaded. FP1/FP2/FP3/Qualifying laps are
never touched. Nothing is fetched from the network when the cache is warm;
the caller is expected to fingerprint the cache directory before and after.

Timing and fuel conventions
---------------------------
Lap time is FastF1's ``LapTime`` converted to seconds. The fuel adjustment
reuses the environment's own model exactly, as documented in
``scripts/analyse_pace_profiles.py``: the environment sets
``fuel_level = 1.0`` at reset, decrements it by ``1/n_laps`` per completed lap
with ``n_laps = 52``, and charges ``2.5 * fuel_level`` seconds per lap. For a
real lap numbered ``L`` (1-indexed) the entry fuel level is therefore
approximated as ``max(0, 1 - (L - 1)/52)`` and

    lap_time_fuel_adj_s = lap_time_s - 2.5 * fuel_level_entry

The linear burn-off is an *assumption*, labelled as such in all outputs.

Estimators
----------
Laps cluster within ``(Driver, Stint)``. Every interval is a cluster
bootstrap over ``(Driver, Stint)`` clusters, resampled with replacement,
percentile 95% CI. Three degradation fits are always reported side by side:
pooled OLS (reproducing ``f1_env.py:233``), a within-stint fixed-effects fit
on cluster-demeaned variables, and a condition-controlled multiple
regression. Divergence between them is the finding, not something to
average away.

Outputs (curated, aggregate only — no record-level CSV is written, so no new
``.gitignore`` pattern is required)
-----------------------------------------------------------------------------
``manifest.csv``            provenance and stage timings
``README.md``               method record
``attrition.csv``           CONSORT-style exclusion cascade (protocol P3)
``weather_classification.csv``  lap classification and agreement counts (P5)
``clip_test.csv``           raw vs clipped slopes, all five compounds (P8)
``estimates.csv``           offsets and slopes with CIs and diagnostics (P7)
``stint_summary.csv``       per-(Driver, Stint) aggregates, in-scope compound
``robustness.csv``          checks R1-R7 (P10)
``thresholds.csv``          the P9 gate, criterion by criterion

Usage
-----
    .venv_f1/bin/python scripts/measure_wet_calibration.py --smoke
    .venv_f1/bin/python scripts/measure_wet_calibration.py --label 20260830T120000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- fixed protocol constants (pre-registered; do not tune) -----------------

LAPS_CSV = Path("data/silverstone_2024_laps.csv")
CACHE_DIR = Path("data/cache")
OUT_ROOT = Path("outputs/phase-04-wet-calibration")

RACE_N_LAPS = 52          # F1RaceEnv default n_laps, and the 2024 British GP distance
FUEL_COEF = 2.5           # f1_env.py:452 / :614
CLIP_LO, CLIP_HI = 0.01, 0.40   # f1_env.py:235
STINT_CLIP_LO, STINT_CLIP_HI = 8, 32   # f1_env.py:242

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
COMPOUND_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

# fallback table, f1_env.py:173-197 — for the "is this value data-derived?" check
FALLBACK = {
    "base_lap_time": 92.0,
    "compound_offsets": {0: -1.2, 1: 0.0, 2: 0.8, 3: 7.0, 4: 11.0},
    "deg_per_lap": {0: 0.18, 1: 0.11, 2: 0.07, 3: 0.20, 4: 0.25},
    "typical_stint": {0: 12, 1: 18, 2: 24, 3: 10, 4: 10},
    "pit_loss": 21.5,
}

# P9 thresholds — approved as written, before any estimate was computed
THRESH_OFFSET = {
    "min_laps": 30,
    "min_clusters": 6,
    "min_drivers": 3,
    "max_ci_halfwidth_s": 2.0,
}
THRESH_SLOPE = {
    "min_laps": 40,
    "min_clusters": 6,
    "min_clusters_with_5plus_laps": 3,
    "min_tyrelife_span": 8,
}

# P5 classification: how many weather samples either side of a Rainfall state
# change are treated as transition
TRANSITION_WINDOW = 2


# --- small helpers ----------------------------------------------------------


def _sec(td: pd.Series) -> pd.Series:
    """Timedelta-ish series -> float seconds."""
    return pd.to_timedelta(td).dt.total_seconds()


def _dir_size_kb(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total // 1024


@dataclass
class Cluster:
    """One (Driver, Stint) group of laps for a single compound."""

    driver: str
    stint: float
    x: np.ndarray          # TyreLife
    y: np.ndarray          # lap time, seconds (raw)
    y_adj: np.ndarray      # lap time, fuel-adjusted
    lapnum: np.ndarray
    wet_class: np.ndarray  # object array of "wet"/"dry"/"mixed"

    @property
    def n(self) -> int:
        return int(self.x.size)

    def key(self) -> Tuple[str, float]:
        return (self.driver, self.stint)


def build_clusters(df: pd.DataFrame, y_col: str = "lap_time_s") -> List[Cluster]:
    out: List[Cluster] = []
    for (drv, stint), g in df.groupby(["Driver", "Stint"], sort=True):
        out.append(
            Cluster(
                driver=str(drv),
                stint=float(stint),
                x=g["TyreLife"].to_numpy(dtype=float),
                y=g[y_col].to_numpy(dtype=float),
                y_adj=g["lap_time_fuel_adj_s"].to_numpy(dtype=float),
                lapnum=g["LapNumber"].to_numpy(dtype=float),
                wet_class=g["wet_class"].to_numpy(dtype=object),
            )
        )
    return out


# --- estimators -------------------------------------------------------------


def _concat(clusters: Sequence[Cluster], adj: bool) -> Tuple[np.ndarray, ...]:
    x = np.concatenate([c.x for c in clusters]) if clusters else np.array([])
    y = np.concatenate([(c.y_adj if adj else c.y) for c in clusters]) if clusters else np.array([])
    ln = np.concatenate([c.lapnum for c in clusters]) if clusters else np.array([])
    wc = np.concatenate([c.wet_class for c in clusters]) if clusters else np.array([], dtype=object)
    return x, y, ln, wc


def pooled_slope(clusters: Sequence[Cluster], adj: bool = False) -> float:
    """OLS slope of lap time on TyreLife — reproduces f1_env.py:233."""
    x, y, _, _ = _concat(clusters, adj)
    if x.size < 2 or np.unique(x).size < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def within_slope(clusters: Sequence[Cluster], adj: bool = False) -> float:
    """Stint fixed-effects slope: OLS on variables demeaned inside each cluster.

    Uses only within-stint variation, so between-stint composition (the
    mechanism behind the documented SOFT artefact) cannot drive it.
    """
    num = 0.0
    den = 0.0
    for c in clusters:
        if c.n < 2:
            continue
        y = c.y_adj if adj else c.y
        xd = c.x - c.x.mean()
        yd = y - y.mean()
        num += float(np.dot(xd, yd))
        den += float(np.dot(xd, xd))
    if den <= 0:
        return float("nan")
    return num / den


def cond_slope(clusters: Sequence[Cluster], adj: bool = False,
               use_wet_dummies: bool = True) -> float:
    """TyreLife coefficient controlling for LapNumber and (if usable) wet class.

    LapNumber proxies elapsed time, hence track drying; the wet-class dummies
    absorb observed condition shifts. Separates degradation from drying only
    to the extent the two are not collinear, which is reported separately.
    """
    x, y, ln, wc = _concat(clusters, adj)
    if x.size < 4 or np.unique(x).size < 2:
        return float("nan")
    cols = [np.ones_like(x), x, ln]
    if use_wet_dummies:
        classes = [c for c in ("wet", "mixed", "dry") if np.sum(wc == c) >= 3]
        # drop the first usable class as the reference level
        for cls in classes[1:]:
            cols.append((wc == cls).astype(float))
    design = np.column_stack(cols)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def median_level(clusters: Sequence[Cluster], adj: bool = False) -> float:
    _, y, _, _ = _concat(clusters, adj)
    return float(np.median(y)) if y.size else float("nan")


def r_squared(clusters: Sequence[Cluster], adj: bool = False) -> float:
    x, y, _, _ = _concat(clusters, adj)
    if x.size < 3 or np.unique(x).size < 2:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def max_cooks_d(clusters: Sequence[Cluster], adj: bool = False) -> float:
    """Max Cook's distance for the pooled simple linear regression."""
    x, y, _, _ = _concat(clusters, adj)
    n = x.size
    if n < 4 or np.unique(x).size < 2:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    p = 2
    mse = float(np.sum(resid ** 2)) / (n - p)
    if mse <= 0:
        return float("nan")
    sxx = float(np.sum((x - x.mean()) ** 2))
    h = 1.0 / n + (x - x.mean()) ** 2 / sxx
    d = (resid ** 2 / (p * mse)) * (h / (1.0 - h) ** 2)
    return float(np.max(d))


# --- cluster bootstrap ------------------------------------------------------


def cluster_bootstrap_ci(
    clusters: Sequence[Cluster],
    stat: Callable[[Sequence[Cluster]], float],
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int]:
    """Percentile 95% CI, resampling whole (Driver, Stint) clusters."""
    k = len(clusters)
    if k == 0:
        return float("nan"), float("nan"), 0
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        vals[b] = stat([clusters[i] for i in idx])
    good = vals[np.isfinite(vals)]
    if good.size < max(20, n_boot // 20):
        return float("nan"), float("nan"), int(good.size)
    lo, hi = np.percentile(good, [2.5, 97.5])
    return float(lo), float(hi), int(good.size)


def paired_diff_ci(
    a: Sequence[Cluster],
    b: Sequence[Cluster],
    stat: Callable[[Sequence[Cluster]], float],
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int]:
    """CI for stat(a) - stat(b), resampling both cluster sets independently.

    Propagates uncertainty in the baseline as well as in the target compound,
    rather than treating the baseline as a known constant.
    """
    ka, kb = len(a), len(b)
    if ka == 0 or kb == 0:
        return float("nan"), float("nan"), 0
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ia = rng.integers(0, ka, size=ka)
        ib = rng.integers(0, kb, size=kb)
        vals[i] = stat([a[j] for j in ia]) - stat([b[j] for j in ib])
    good = vals[np.isfinite(vals)]
    if good.size < max(20, n_boot // 20):
        return float("nan"), float("nan"), int(good.size)
    lo, hi = np.percentile(good, [2.5, 97.5])
    return float(lo), float(hi), int(good.size)


# --- data assembly ----------------------------------------------------------


@dataclass
class Stage:
    name: str
    seconds: float


@dataclass
class Assembly:
    csv_raw: pd.DataFrame
    merged: pd.DataFrame
    attrition: List[dict] = field(default_factory=list)
    weather_rows: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(LAPS_CSV)
    df["lap_time_s"] = _sec(df["LapTime"])
    df["Compound"] = df["Compound"].astype(str).str.upper()
    df["fuel_level_entry"] = np.maximum(
        0.0, 1.0 - (df["LapNumber"].astype(float) - 1.0) / RACE_N_LAPS
    )
    df["lap_time_fuel_adj_s"] = df["lap_time_s"] - FUEL_COEF * df["fuel_level_entry"]
    return df


def load_session_aux() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the already-cached 2024 British GP Race session, read-only."""
    import fastf1

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(2024, "Silverstone", "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(laps=True, telemetry=False, weather=True, messages=True)

    laps = session.laps.copy()
    aux_cols = ["Driver", "LapNumber", "LapStartTime", "Time", "TrackStatus",
                "PitInTime", "PitOutTime", "Position", "IsAccurate",
                "Compound", "TyreLife", "Stint", "FreshTyre"]
    if "Deleted" in laps.columns:
        aux_cols.append("Deleted")
    if "DeletedReason" in laps.columns:
        aux_cols.append("DeletedReason")
    aux = laps[[c for c in aux_cols if c in laps.columns]].copy()

    weather = session.weather_data.copy()
    meta = {
        "session_laps_rows": int(len(laps)),
        "session_weather_rows": int(len(weather)),
        "has_deleted_col": bool("Deleted" in laps.columns),
        "event": str(session.event.get("EventName", "")),
        "session_name": str(getattr(session, "name", "")),
    }
    return aux, weather, meta


def classify_weather(merged: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Protocol P5: per-lap wet / dry / mixed from the Rainfall boolean.

    Session is treated as time-varying. Rainfall is the sole classifying
    signal; TrackTemp and Compound corroborate and are recorded, never used
    to override.
    """
    w = weather.copy()
    w["t"] = _sec(w["Time"])
    w = w.sort_values("t").reset_index(drop=True)
    rain = w["Rainfall"].astype(bool).to_numpy()

    # indices adjacent to a Rainfall state change, widened by TRANSITION_WINDOW
    change_idx = np.flatnonzero(np.diff(rain.astype(int)) != 0)
    transition = np.zeros(len(w), dtype=bool)
    for ci in change_idx:
        lo = max(0, ci - TRANSITION_WINDOW + 1)
        hi = min(len(w) - 1, ci + TRANSITION_WINDOW)
        transition[lo:hi + 1] = True

    wt = w["t"].to_numpy()
    start = merged["lap_start_s"].to_numpy(dtype=float)
    dur = merged["lap_time_s"].to_numpy(dtype=float)

    classes: List[str] = []
    join_gaps: List[float] = []
    n_cov: List[int] = []
    trktemp: List[float] = []
    airtemp: List[float] = []
    any_trans: List[bool] = []

    tt = w["TrackTemp"].to_numpy(dtype=float)
    at = w["AirTemp"].to_numpy(dtype=float)

    for s, d in zip(start, dur):
        if not np.isfinite(s):
            classes.append("unknown")
            join_gaps.append(float("nan"))
            n_cov.append(0)
            trktemp.append(float("nan"))
            airtemp.append(float("nan"))
            any_trans.append(False)
            continue
        end = s + (d if np.isfinite(d) else 0.0)
        # nearest sample at or before lap start
        prev = np.searchsorted(wt, s, side="right") - 1
        prev = max(prev, 0)
        inside = np.flatnonzero((wt >= s) & (wt <= end))
        sel = np.unique(np.concatenate(([prev], inside))) if inside.size else np.array([prev])
        vals = rain[sel]
        in_trans = bool(np.any(transition[sel]))
        if in_trans or (vals.any() and not vals.all()):
            cls = "mixed"
        elif vals.all():
            cls = "wet"
        else:
            cls = "dry"
        classes.append(cls)
        join_gaps.append(float(s - wt[prev]))
        n_cov.append(int(sel.size))
        trktemp.append(float(np.mean(tt[sel])))
        airtemp.append(float(np.mean(at[sel])))
        any_trans.append(in_trans)

    merged = merged.copy()
    merged["wet_class"] = classes
    merged["weather_join_gap_s"] = join_gaps
    merged["n_weather_samples"] = n_cov
    merged["track_temp_c"] = trktemp
    merged["air_temp_c"] = airtemp
    merged["in_transition"] = any_trans
    return merged


def assemble(smoke: bool) -> Tuple[Assembly, dict, List[Stage]]:
    stages: List[Stage] = []

    t0 = time.perf_counter()
    csv_raw = load_csv()
    stages.append(Stage("load_csv", time.perf_counter() - t0))

    t0 = time.perf_counter()
    aux, weather, meta = load_session_aux()
    stages.append(Stage("load_cached_race_session", time.perf_counter() - t0))

    t0 = time.perf_counter()
    # provenance: the CSV must be the IsAccurate subset of this session
    aux_key = aux[["Driver", "LapNumber"]].copy()
    aux_key["LapNumber"] = aux_key["LapNumber"].astype(float)
    csv_key = csv_raw[["Driver", "LapNumber"]].copy()
    csv_key["LapNumber"] = csv_key["LapNumber"].astype(float)

    aux2 = aux.copy()
    aux2["LapNumber"] = aux2["LapNumber"].astype(float)
    aux2 = aux2.rename(columns={
        "Compound": "Compound_sess", "TyreLife": "TyreLife_sess",
        "Stint": "Stint_sess", "FreshTyre": "FreshTyre_sess",
        "IsAccurate": "IsAccurate_sess", "TrackStatus": "TrackStatus_sess",
    })
    merged = csv_raw.merge(aux2, on=["Driver", "LapNumber"], how="left",
                           validate="one_to_one")
    merged["lap_start_s"] = _sec(merged["LapStartTime"])

    notes: List[str] = []
    n_unmatched = int(merged["lap_start_s"].isna().sum())
    notes.append(f"CSV rows unmatched against the cached session: {n_unmatched}")
    for col, sess_col in (("Compound", "Compound_sess"),
                          ("TyreLife", "TyreLife_sess"),
                          ("Stint", "Stint_sess")):
        if sess_col in merged.columns:
            a = merged[col]
            b = merged[sess_col]
            if col == "Compound":
                mism = int((a.astype(str).str.upper() != b.astype(str).str.upper()).sum())
            else:
                mism = int((a.astype(float) != b.astype(float)).sum())
            notes.append(f"{col} mismatches CSV vs session: {mism}")

    merged = classify_weather(merged, weather)
    stages.append(Stage("merge_and_classify", time.perf_counter() - t0))

    asm = Assembly(csv_raw=csv_raw, merged=merged, notes=notes)
    asm.attrition.append({"step": "0_csv_rows", "rule": "data/silverstone_2024_laps.csv as delivered",
                          "rows": int(len(csv_raw)), "dropped": 0})
    meta["n_unmatched"] = n_unmatched
    meta["smoke"] = smoke
    return asm, meta, stages


# --- exclusion cascade (protocol P3) ---------------------------------------


def apply_exclusions(merged: pd.DataFrame, attrition: List[dict],
                     relax: bool = False) -> pd.DataFrame:
    """P3 rules 1-8. ``relax=True`` is robustness check R1 (rules 4-6 off)."""
    df = merged.copy()
    prev = len(df)

    def step(name: str, rule: str, d: pd.DataFrame) -> pd.DataFrame:
        nonlocal prev
        if not relax:
            attrition.append({"step": name, "rule": rule, "rows": int(len(d)),
                              "dropped": int(prev - len(d))})
        prev = len(d)
        return d

    df = step("1_isaccurate", "IsAccurate == True (already applied upstream)",
              df[df["IsAccurate"].astype(bool)])
    df = step("2_nulls", "drop null LapTime/Compound/TyreLife, non-finite seconds",
              df.dropna(subset=["LapTime", "Compound", "TyreLife"])
                .loc[lambda d: np.isfinite(d["lap_time_s"])])

    if not relax:
        if "Deleted" in df.columns:
            mask = df["Deleted"].fillna(False).astype(bool)
            df = step("3_deleted", "drop Deleted == True (not caught by IsAccurate)",
                      df[~mask])
        else:
            attrition.append({"step": "3_deleted", "rule": "Deleted column unavailable — rule skipped",
                              "rows": int(len(df)), "dropped": 0})

        ts = df["TrackStatus_sess"].astype(str) if "TrackStatus_sess" in df.columns \
            else df["TrackStatus"].astype(str)
        df = step("4_trackstatus", 'drop TrackStatus != "1" (assert, do not assume)',
                  df[ts == "1"])

        # rule 6: drop the first green lap after any SC/VSC period
        drop_idx: List[int] = []
        for drv, g in df.groupby("Driver"):
            g = g.sort_values("LapNumber")
            lapnums = g["LapNumber"].to_numpy(dtype=float)
            # SC/VSC laps are already gone; detect a gap in lap numbering that
            # spans a removed non-green lap, and drop the lap resuming after it
            for i in range(1, len(lapnums)):
                if lapnums[i] - lapnums[i - 1] > 1:
                    drop_idx.append(int(g.index[i]))
        df = step("5_post_sc_restart",
                  "drop first surviving lap after any gap left by a non-green lap",
                  df.drop(index=drop_idx, errors="ignore"))

    sizes = df.groupby(["Driver", "Stint"]).size()
    keep = sizes[sizes >= 3].index
    df = df.set_index(["Driver", "Stint"])
    df = df[df.index.isin(keep)].reset_index()
    if not relax:
        attrition.append({"step": "6_min_cluster",
                          "rule": "drop (Driver, Stint) clusters with < 3 surviving laps",
                          "rows": int(len(df)), "dropped": int(prev - len(df))})
    return df


# --- P8 clip test -----------------------------------------------------------


def clip_test(csv_raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce f1_env.py:202-254 exactly, but expose the unclipped slope."""
    df = csv_raw.dropna(subset=["LapTime", "Compound", "TyreLife"]).copy()
    df = df[np.isfinite(df["lap_time_s"])]
    dry = df[df["Compound"].isin(list(DRY_COMPOUNDS))]
    base_lap = float(dry["lap_time_s"].median())

    rows = []
    for name, idx in COMPOUND_TO_IDX.items():
        cdf = df[df["Compound"] == name]
        n = int(len(cdf))
        if n == 0:
            rows.append({
                "compound": name, "idx": idx, "n_laps": 0,
                "raw_slope": None, "clipped_slope": FALLBACK["deg_per_lap"][idx],
                "clip_active": None, "clip_direction": "n/a — fallback used",
                "offset_vs_base": FALLBACK["compound_offsets"][idx],
                "typical_stint": FALLBACK["typical_stint"][idx],
                "stint_clip_active": None,
                "source": "fallback (no laps)",
            })
            continue
        raw = float(np.polyfit(cdf["TyreLife"].astype(float).values,
                               cdf["lap_time_s"].astype(float).values, 1)[0]) \
            if cdf["TyreLife"].nunique() > 1 else float("nan")
        clipped = float(np.clip(raw, CLIP_LO, CLIP_HI)) if np.isfinite(raw) else float("nan")
        stint_lengths = cdf.groupby(["Driver", "Stint"]).size()
        raw_stint = float(stint_lengths.median()) if len(stint_lengths) else float("nan")
        clipped_stint = int(np.clip(raw_stint, STINT_CLIP_LO, STINT_CLIP_HI)) \
            if np.isfinite(raw_stint) else None
        direction = "not active"
        if np.isfinite(raw):
            if raw < CLIP_LO:
                direction = f"FLOOR active (raw {raw:.6f} < {CLIP_LO})"
            elif raw > CLIP_HI:
                direction = f"CEIL active (raw {raw:.6f} > {CLIP_HI})"
        rows.append({
            "compound": name, "idx": idx, "n_laps": n,
            "raw_slope": raw, "clipped_slope": clipped,
            "clip_active": bool(np.isfinite(raw) and (raw < CLIP_LO or raw > CLIP_HI)),
            "clip_direction": direction,
            "offset_vs_base": float(cdf["lap_time_s"].median() - base_lap),
            "raw_median_stint": raw_stint,
            "typical_stint": clipped_stint,
            "stint_clip_active": bool(np.isfinite(raw_stint)
                                      and (raw_stint < STINT_CLIP_LO
                                           or raw_stint > STINT_CLIP_HI)),
            "source": "data-derived",
        })
    out = pd.DataFrame(rows)
    out.attrs["base_lap_time"] = base_lap
    return out


# --- main measurement -------------------------------------------------------


def measure(target: str, df: pd.DataFrame, n_boot: int,
            rng: np.random.Generator) -> Tuple[List[dict], List[dict]]:
    """Offsets and slopes for ``target`` with cluster-bootstrap CIs."""
    est: List[dict] = []

    tgt = df[df["Compound"] == target]
    tgt_cl = build_clusters(tgt)

    # --- baselines (protocol P2) ---
    dry_all = df[df["Compound"].isin(list(DRY_COMPOUNDS))]
    dry_green = dry_all[dry_all["wet_class"] == "dry"]
    med = df[df["Compound"] == "MEDIUM"]

    baselines = {
        "B_current (median of all dry laps — reproduces live code)": dry_all,
        "B_dry_green (median of dry laps classified dry by P5)": dry_green,
        "B_medium (median of MEDIUM laps only)": med,
    }

    for bname, bdf in baselines.items():
        bcl = build_clusters(bdf)
        point = median_level(tgt_cl) - median_level(bcl)
        lo, hi, nb = paired_diff_ci(tgt_cl, bcl, median_level, n_boot, rng)
        est.append({
            "quantity": "compound_offset_s",
            "compound": target,
            "estimator": "median difference",
            "baseline": bname,
            "point": point,
            "ci_lo": lo, "ci_hi": hi,
            "ci_halfwidth": (hi - lo) / 2 if np.isfinite(lo) and np.isfinite(hi) else float("nan"),
            "n_laps": int(len(tgt)),
            "n_clusters": len(tgt_cl),
            "n_drivers": int(tgt["Driver"].nunique()),
            "n_baseline_laps": int(len(bdf)),
            "n_baseline_clusters": len(bcl),
            "boot_ok": nb,
            "label": "derived",
        })

    # --- degradation slopes (protocol P7), raw and fuel-adjusted ---
    span = float(tgt["TyreLife"].max() - tgt["TyreLife"].min()) if len(tgt) else float("nan")
    sizes = tgt.groupby(["Driver", "Stint"]).size()
    fits = [
        ("pooled OLS (reproduces f1_env.py:233)", pooled_slope),
        ("within-stint fixed effects", within_slope),
        ("condition-controlled (+LapNumber, +wet class)", cond_slope),
    ]
    for adj in (False, True):
        for fname, fn in fits:
            def stat(cs, _fn=fn, _adj=adj):
                return _fn(cs, adj=_adj)
            point = stat(tgt_cl)
            lo, hi, nb = cluster_bootstrap_ci(tgt_cl, stat, n_boot, rng)
            est.append({
                "quantity": "deg_per_lap_s_per_lap",
                "compound": target,
                "estimator": fname,
                "baseline": "fuel-adjusted" if adj else "raw lap time",
                "point": point,
                "ci_lo": lo, "ci_hi": hi,
                "ci_halfwidth": (hi - lo) / 2 if np.isfinite(lo) and np.isfinite(hi) else float("nan"),
                "n_laps": int(len(tgt)),
                "n_clusters": len(tgt_cl),
                "n_drivers": int(tgt["Driver"].nunique()),
                "tyrelife_min": float(tgt["TyreLife"].min()) if len(tgt) else float("nan"),
                "tyrelife_max": float(tgt["TyreLife"].max()) if len(tgt) else float("nan"),
                "tyrelife_span": span,
                "n_clusters_5plus": int((sizes >= 5).sum()),
                "r_squared": r_squared(tgt_cl, adj=adj) if "pooled" in fname else float("nan"),
                "max_cooks_d": max_cooks_d(tgt_cl, adj=adj) if "pooled" in fname else float("nan"),
                "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                         and (lo > 0 or hi < 0)),
                "boot_ok": nb,
                "label": "derived",
            })

    # --- per-cluster summary (aggregate, not record-level) ---
    stint_rows: List[dict] = []
    for c in tgt_cl:
        b = float(np.polyfit(c.x, c.y, 1)[0]) if (c.n >= 2 and np.unique(c.x).size > 1) else float("nan")
        stint_rows.append({
            "compound": target,
            "driver": c.driver,
            "stint": c.stint,
            "n_laps": c.n,
            "tyrelife_min": float(c.x.min()),
            "tyrelife_max": float(c.x.max()),
            "lapnum_min": float(c.lapnum.min()),
            "lapnum_max": float(c.lapnum.max()),
            "median_lap_time_s": float(np.median(c.y)),
            "median_lap_time_fuel_adj_s": float(np.median(c.y_adj)),
            "within_stint_slope": b,
            "wet_frac": float(np.mean(c.wet_class == "wet")),
            "mixed_frac": float(np.mean(c.wet_class == "mixed")),
            "dry_frac": float(np.mean(c.wet_class == "dry")),
        })
    return est, stint_rows


def robustness(target: str, merged: pd.DataFrame, df: pd.DataFrame,
               n_boot: int, rng: np.random.Generator) -> List[dict]:
    """Protocol P10, checks R1-R7."""
    rows: List[dict] = []
    boot_r = max(2000, n_boot // 5)   # robustness CIs use a lighter bootstrap

    def record(rid: str, desc: str, sub: pd.DataFrame, compound: str = None):
        comp = compound or target
        t = sub[sub["Compound"] == comp]
        cl = build_clusters(t)
        if len(cl) == 0:
            rows.append({"check": rid, "description": desc, "compound": comp,
                         "n_laps": 0, "n_clusters": 0,
                         "pooled_slope": None, "within_slope": None,
                         "cond_slope": None, "within_ci_lo": None,
                         "within_ci_hi": None, "median_lap_time_s": None,
                         "sign_agreement": None})
            return
        ps = pooled_slope(cl)
        ws = within_slope(cl)
        cs = cond_slope(cl)
        lo, hi, _ = cluster_bootstrap_ci(cl, lambda c: within_slope(c), boot_r, rng)
        rows.append({
            "check": rid, "description": desc, "compound": comp,
            "n_laps": int(len(t)), "n_clusters": len(cl),
            "pooled_slope": ps, "within_slope": ws, "cond_slope": cs,
            "within_ci_lo": lo, "within_ci_hi": hi,
            "median_lap_time_s": float(np.median(np.concatenate([c.y for c in cl]))),
            "sign_agreement": bool(np.isfinite(ps) and np.isfinite(ws)
                                   and np.sign(ps) == np.sign(ws)),
        })

    record("R0_primary", "primary specification (all P3 rules applied)", df)

    attr_dummy: List[dict] = []
    relaxed = apply_exclusions(merged, attr_dummy, relax=True)
    record("R1_relaxed_exclusions", "P3 rules 4-6 removed (live-path filtering only)", relaxed)

    # R2 handled separately below (location estimator, not a slope)
    r3 = df[df["FreshTyre"].astype(str).str.lower() == "true"]
    record("R3_fresh_only", "exclude used sets (FreshTyre == False)", r3)

    # R4 leave-one-driver-out and leave-one-stint-out extremes
    tgt = df[df["Compound"] == target]
    cl_all = build_clusters(tgt)
    if len(cl_all) > 2:
        lodo = []
        for drv in sorted(tgt["Driver"].unique()):
            sub = [c for c in cl_all if c.driver != drv]
            if len(sub) >= 2:
                lodo.append((drv, within_slope(sub), pooled_slope(sub)))
        if lodo:
            ws_vals = [v for _, v, _ in lodo if np.isfinite(v)]
            ps_vals = [v for _, _, v in lodo if np.isfinite(v)]
            rows.append({
                "check": "R4_leave_one_driver_out",
                "description": (f"{len(lodo)} refits, one driver removed each; "
                                "range across refits"),
                "compound": target, "n_laps": int(len(tgt)), "n_clusters": len(cl_all),
                "pooled_slope": f"[{min(ps_vals):.6f}, {max(ps_vals):.6f}]" if ps_vals else None,
                "within_slope": f"[{min(ws_vals):.6f}, {max(ws_vals):.6f}]" if ws_vals else None,
                "cond_slope": None, "within_ci_lo": None, "within_ci_hi": None,
                "median_lap_time_s": None,
                "sign_agreement": bool(ws_vals and (min(ws_vals) > 0 or max(ws_vals) < 0)),
            })
        loso = []
        for c_drop in cl_all:
            sub = [c for c in cl_all if c.key() != c_drop.key()]
            if len(sub) >= 2:
                loso.append((c_drop.key(), within_slope(sub), pooled_slope(sub)))
        if loso:
            ws_vals = [v for _, v, _ in loso if np.isfinite(v)]
            ps_vals = [v for _, _, v in loso if np.isfinite(v)]
            rows.append({
                "check": "R4_leave_one_stint_out",
                "description": (f"{len(loso)} refits, one (Driver, Stint) removed each; "
                                "range across refits"),
                "compound": target, "n_laps": int(len(tgt)), "n_clusters": len(cl_all),
                "pooled_slope": f"[{min(ps_vals):.6f}, {max(ps_vals):.6f}]" if ps_vals else None,
                "within_slope": f"[{min(ws_vals):.6f}, {max(ws_vals):.6f}]" if ws_vals else None,
                "cond_slope": None, "within_ci_lo": None, "within_ci_hi": None,
                "median_lap_time_s": None,
                "sign_agreement": bool(ws_vals and (min(ws_vals) > 0 or max(ws_vals) < 0)),
            })

    record("R5_wet_only", "laps classified wet only (mixed/transition excluded)",
           df[df["wet_class"] == "wet"])

    # R6: fuel-adjusted variant of the primary fit
    if len(cl_all):
        ps = pooled_slope(cl_all, adj=True)
        ws = within_slope(cl_all, adj=True)
        cs = cond_slope(cl_all, adj=True)
        lo, hi, _ = cluster_bootstrap_ci(cl_all, lambda c: within_slope(c, adj=True),
                                         boot_r, rng)
        rows.append({
            "check": "R6_fuel_adjusted",
            "description": "primary specification on fuel-adjusted lap time",
            "compound": target, "n_laps": int(len(tgt)), "n_clusters": len(cl_all),
            "pooled_slope": ps, "within_slope": ws, "cond_slope": cs,
            "within_ci_lo": lo, "within_ci_hi": hi,
            "median_lap_time_s": float(np.median(np.concatenate([c.y_adj for c in cl_all]))),
            "sign_agreement": bool(np.isfinite(ps) and np.isfinite(ws)
                                   and np.sign(ps) == np.sign(ws)),
        })

    # R7: falsification — does the protocol reproduce the known SOFT artefact?
    for comp in ("SOFT", "MEDIUM", "HARD"):
        record(f"R7_{comp.lower()}", f"falsification: same three fits on {comp}",
               df, compound=comp)

    # R2: location-estimator sensitivity for the offset (not a slope check)
    dry_all = build_clusters(df[df["Compound"].isin(list(DRY_COMPOUNDS))])
    if len(cl_all) and len(dry_all):
        _, y_t, _, _ = _concat(cl_all, False)
        _, y_b, _, _ = _concat(dry_all, False)
        for est_name, fn in (("median", lambda a: float(np.median(a))),
                             ("trimmed mean 10%", lambda a: trimmed(a, 0.1)),
                             ("mean", lambda a: float(np.mean(a)))):
            rows.append({
                "check": f"R2_offset_{est_name.split()[0]}",
                "description": (f"offset vs B_current using {est_name} "
                                "(location estimator sensitivity)"),
                "compound": target,
                "n_laps": int(y_t.size), "n_clusters": len(cl_all),
                "pooled_slope": None, "within_slope": None, "cond_slope": None,
                "within_ci_lo": None, "within_ci_hi": None,
                "median_lap_time_s": fn(y_t) - fn(y_b),
                "sign_agreement": None,
            })
    return rows


def trimmed(a: np.ndarray, frac: float = 0.1) -> float:
    if a.size == 0:
        return float("nan")
    k = int(np.floor(a.size * frac))
    s = np.sort(a)
    if 2 * k >= s.size:
        return float(np.median(s))
    return float(np.mean(s[k:s.size - k]))


def gate(target: str, est: List[dict], clip: pd.DataFrame,
         df: pd.DataFrame) -> List[dict]:
    """Protocol P9 gate, criterion by criterion."""
    tgt = df[df["Compound"] == target]
    sizes = tgt.groupby(["Driver", "Stint"]).size()
    n_laps = int(len(tgt))
    n_clusters = int(len(sizes))
    n_drivers = int(tgt["Driver"].nunique())
    span = float(tgt["TyreLife"].max() - tgt["TyreLife"].min()) if n_laps else float("nan")

    off = [e for e in est if e["quantity"] == "compound_offset_s"
           and e["baseline"].startswith("B_current")]
    off_hw = off[0]["ci_halfwidth"] if off else float("nan")

    pooled = [e for e in est if e["quantity"] == "deg_per_lap_s_per_lap"
              and "pooled" in e["estimator"] and e["baseline"] == "raw lap time"]
    within = [e for e in est if e["quantity"] == "deg_per_lap_s_per_lap"
              and "within" in e["estimator"] and e["baseline"] == "raw lap time"]
    p_pt = pooled[0]["point"] if pooled else float("nan")
    w_pt = within[0]["point"] if within else float("nan")
    w_excl0 = within[0]["ci_excludes_zero"] if within else False

    crit: List[dict] = []

    def add(group: str, name: str, req, obs, ok: bool):
        crit.append({"group": group, "criterion": name, "required": str(req),
                     "observed": str(obs), "pass": bool(ok)})

    add("offset", "n_laps >= 30", 30, n_laps, n_laps >= THRESH_OFFSET["min_laps"])
    add("offset", "n_clusters >= 6", 6, n_clusters, n_clusters >= THRESH_OFFSET["min_clusters"])
    add("offset", "n_drivers >= 3", 3, n_drivers, n_drivers >= THRESH_OFFSET["min_drivers"])
    add("offset", "CI halfwidth <= 2.0 s", "<= 2.0",
        f"{off_hw:.4f}" if np.isfinite(off_hw) else "nan",
        bool(np.isfinite(off_hw) and off_hw <= THRESH_OFFSET["max_ci_halfwidth_s"]))

    add("slope", "n_laps >= 40", 40, n_laps, n_laps >= THRESH_SLOPE["min_laps"])
    add("slope", "n_clusters >= 6", 6, n_clusters, n_clusters >= THRESH_SLOPE["min_clusters"])
    add("slope", "clusters with >= 5 laps >= 3", 3, int((sizes >= 5).sum()),
        int((sizes >= 5).sum()) >= THRESH_SLOPE["min_clusters_with_5plus_laps"])
    add("slope", "TyreLife span >= 8", 8, span,
        bool(np.isfinite(span) and span >= THRESH_SLOPE["min_tyrelife_span"]))
    add("slope", "within-stint CI excludes 0", True, w_excl0, bool(w_excl0))
    add("slope", "pooled and within-stint slopes agree in sign", True,
        f"pooled {p_pt:.6f} / within {w_pt:.6f}",
        bool(np.isfinite(p_pt) and np.isfinite(w_pt) and np.sign(p_pt) == np.sign(w_pt)))

    row = clip[clip["compound"] == target]
    raw = float(row["raw_slope"].iloc[0]) if len(row) and row["raw_slope"].iloc[0] is not None else float("nan")
    add("slope", "raw (unclipped) slope not negative and not ~0", "> 0 and CI excludes 0",
        f"raw {raw:.6f}" if np.isfinite(raw) else "nan",
        bool(np.isfinite(raw) and raw > 0 and w_excl0))
    return crit


# --- entrypoint -------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=None, help="output subdirectory name")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="cluster-bootstrap resamples (protocol: 10000)")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scope schema check: 200 resamples, label smoke_test")
    args = ap.parse_args(argv)

    if args.smoke:
        args.bootstrap = min(args.bootstrap, 200)
        args.label = args.label or "smoke_test"
    label = args.label or time.strftime("%Y%m%dT%H%M%S")
    outdir = OUT_ROOT / label
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cache_before = _dir_size_kb(CACHE_DIR)
    t_start = time.perf_counter()

    asm, meta, stages = assemble(args.smoke)

    t0 = time.perf_counter()
    clip = clip_test(asm.csv_raw)
    stages.append(Stage("clip_test", time.perf_counter() - t0))

    t0 = time.perf_counter()
    df = apply_exclusions(asm.merged, asm.attrition)
    stages.append(Stage("exclusions", time.perf_counter() - t0))

    # weather classification summary
    wrows: List[dict] = []
    for comp in ["ALL"] + list(COMPOUND_TO_IDX):
        sub = asm.merged if comp == "ALL" else asm.merged[asm.merged["Compound"] == comp]
        if not len(sub):
            continue
        wrows.append({
            "scope": comp,
            "n_laps": int(len(sub)),
            "n_wet": int((sub["wet_class"] == "wet").sum()),
            "n_mixed": int((sub["wet_class"] == "mixed").sum()),
            "n_dry": int((sub["wet_class"] == "dry").sum()),
            "n_unknown": int((sub["wet_class"] == "unknown").sum()),
            "median_track_temp_c": float(sub["track_temp_c"].median()),
            "median_air_temp_c": float(sub["air_temp_c"].median()),
            "max_weather_join_gap_s": float(sub["weather_join_gap_s"].max()),
            "median_weather_join_gap_s": float(sub["weather_join_gap_s"].median()),
        })
    # agreement: Rainfall class vs compound choice
    inter = asm.merged[asm.merged["Compound"] == "INTERMEDIATE"]
    dryc = asm.merged[asm.merged["Compound"].isin(list(DRY_COMPOUNDS))]
    wrows.append({
        "scope": "AGREEMENT_intermediate_on_dry_classified_laps",
        "n_laps": int(len(inter)),
        "n_wet": int((inter["wet_class"] == "wet").sum()),
        "n_mixed": int((inter["wet_class"] == "mixed").sum()),
        "n_dry": int((inter["wet_class"] == "dry").sum()),
        "n_unknown": 0,
        "median_track_temp_c": float("nan"), "median_air_temp_c": float("nan"),
        "max_weather_join_gap_s": float("nan"),
        "median_weather_join_gap_s": float("nan"),
    })
    wrows.append({
        "scope": "AGREEMENT_dry_compound_on_wet_classified_laps",
        "n_laps": int(len(dryc)),
        "n_wet": int((dryc["wet_class"] == "wet").sum()),
        "n_mixed": int((dryc["wet_class"] == "mixed").sum()),
        "n_dry": int((dryc["wet_class"] == "dry").sum()),
        "n_unknown": 0,
        "median_track_temp_c": float("nan"), "median_air_temp_c": float("nan"),
        "max_weather_join_gap_s": float("nan"),
        "median_weather_join_gap_s": float("nan"),
    })

    t0 = time.perf_counter()
    est, stint_rows = measure("INTERMEDIATE", df, args.bootstrap, rng)
    stages.append(Stage("estimates_intermediate", time.perf_counter() - t0))

    # WET: pre-determined no-calibration outcome, recorded explicitly
    n_wet_laps = int((asm.csv_raw["Compound"] == "WET").sum())
    est.append({
        "quantity": "compound_offset_s", "compound": "WET",
        "estimator": "not estimated — zero laps in the calibration source",
        "baseline": "n/a", "point": float("nan"), "ci_lo": float("nan"),
        "ci_hi": float("nan"), "ci_halfwidth": float("nan"),
        "n_laps": n_wet_laps, "n_clusters": 0, "n_drivers": 0,
        "label": "assumption (fallback 11.0 s, unevidenced)",
    })
    est.append({
        "quantity": "deg_per_lap_s_per_lap", "compound": "WET",
        "estimator": "not estimated — zero laps in the calibration source",
        "baseline": "n/a", "point": float("nan"), "ci_lo": float("nan"),
        "ci_hi": float("nan"), "ci_halfwidth": float("nan"),
        "n_laps": n_wet_laps, "n_clusters": 0, "n_drivers": 0,
        "label": "assumption (fallback 0.25 s/lap, unevidenced)",
    })

    t0 = time.perf_counter()
    rob = robustness("INTERMEDIATE", asm.merged, df, args.bootstrap, rng)
    stages.append(Stage("robustness", time.perf_counter() - t0))

    t0 = time.perf_counter()
    crit = gate("INTERMEDIATE", est, clip, df)
    stages.append(Stage("gate", time.perf_counter() - t0))

    # --- write curated outputs ---
    pd.DataFrame(asm.attrition).to_csv(outdir / "attrition.csv", index=False)
    pd.DataFrame(wrows).to_csv(outdir / "weather_classification.csv", index=False)
    clip.to_csv(outdir / "clip_test.csv", index=False)
    pd.DataFrame(est).to_csv(outdir / "estimates.csv", index=False)
    pd.DataFrame(stint_rows).to_csv(outdir / "stint_summary.csv", index=False)
    pd.DataFrame(rob).to_csv(outdir / "robustness.csv", index=False)
    pd.DataFrame(crit).to_csv(outdir / "thresholds.csv", index=False)

    cache_after = _dir_size_kb(CACHE_DIR)
    total = time.perf_counter() - t_start
    manifest = {
        "script": "scripts/measure_wet_calibration.py",
        "label": label,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "smoke": args.smoke,
        "calibration_source": str(LAPS_CSV),
        "calibration_source_rows": int(len(asm.csv_raw)),
        "aux_session": "2024 British Grand Prix — Race (cached, read-only)",
        "sessions_loaded": "R only; FP1/FP2/FP3/Q never loaded",
        "cache_dir_kb_before": cache_before,
        "cache_dir_kb_after": cache_after,
        "cache_growth_kb": (cache_after - cache_before)
        if (cache_before is not None and cache_after is not None) else None,
        "base_lap_time_reproduced": clip.attrs.get("base_lap_time"),
        "total_seconds": round(total, 3),
        "provenance_notes": " | ".join(asm.notes),
    }
    manifest.update({f"meta_{k}": v for k, v in meta.items()})
    manifest.update({f"stage_{s.name}_s": round(s.seconds, 3) for s in stages})
    pd.DataFrame([manifest]).to_csv(outdir / "manifest.csv", index=False)

    (outdir / "README.md").write_text(_readme(label, manifest, crit), encoding="utf-8")

    # --- console summary ---
    print(f"\n=== Phase 04 measurement — label {label} ===")
    print(f"output: {outdir}")
    print(f"cache growth: {manifest['cache_growth_kb']} KB")
    print(f"\n--- attrition ---")
    print(pd.DataFrame(asm.attrition).to_string(index=False))
    print(f"\n--- P8 clip test ---")
    print(clip[["compound", "n_laps", "raw_slope", "clipped_slope",
                "clip_active", "clip_direction", "offset_vs_base"]].to_string(index=False))
    print(f"\n--- weather classification ---")
    print(pd.DataFrame(wrows).to_string(index=False))
    print(f"\n--- estimates ---")
    print(pd.DataFrame(est).to_string(index=False))
    print(f"\n--- robustness ---")
    print(pd.DataFrame(rob).to_string(index=False))
    print(f"\n--- P9 gate ---")
    print(pd.DataFrame(crit).to_string(index=False))
    n_fail = sum(1 for c in crit if not c["pass"])
    print(f"\nP9: {len(crit) - n_fail}/{len(crit)} criteria passed; {n_fail} failed")
    print("provenance:", manifest["provenance_notes"])
    return 0


def _readme(label: str, manifest: dict, crit: List[dict]) -> str:
    n_fail = sum(1 for c in crit if not c["pass"])
    return f"""# Phase 04 wet-session calibration — measurement outputs (`{label}`)

Generated by `scripts/measure_wet_calibration.py`, a **read-only** measurement
script. Nothing in the environment, reward, hazard, tyre, training, evaluator or
model code was executed or modified to produce these files. `F1RaceEnv` is not
imported by this script.

## Provenance

| Item | Value |
|---|---|
| Calibration source | `{manifest['calibration_source']}` ({manifest['calibration_source_rows']} rows) |
| Auxiliary session | {manifest['aux_session']} |
| Sessions loaded | {manifest['sessions_loaded']} |
| Cache growth | {manifest['cache_growth_kb']} KB |
| Bootstrap resamples | {manifest['bootstrap_resamples']} (cluster bootstrap over `(Driver, Stint)`) |
| Seed | {manifest['seed']} |
| Reproduced `base_lap_time` | {manifest['base_lap_time_reproduced']} |
| P9 gate | {len(crit) - n_fail}/{len(crit)} criteria passed |

## Method

Protocol as pre-registered in `docs/phase-04-wet-session-evidence.md`
(sections P0-P11), approved before any estimate was computed.

**Timing.** `LapTime` in seconds. The fuel adjustment reuses the
environment's own model: `fuel_level_entry = max(0, 1 - (LapNumber - 1)/52)`
and `lap_time_fuel_adj_s = lap_time_s - 2.5 * fuel_level_entry`, matching
`f1_env.py:452` and the convention documented in
`scripts/analyse_pace_profiles.py`. Linear burn-off is an **assumption**.

**Clustering.** Laps are not independent; they cluster within
`(Driver, Stint)`. Every confidence interval resamples whole clusters with
replacement (percentile 95%).

**Three slope fits, always reported together.** Pooled OLS reproduces
`f1_env.py:233` exactly. The within-stint fit demeans both variables inside
each cluster, so only within-stint variation contributes and between-stint
composition cannot drive the estimate. The condition-controlled fit adds
`LapNumber` and wet-class dummies. Divergence between the three is the
finding, not noise to be averaged.

**Weather.** The session is treated as **time-varying**, never as a single
condition. `Rainfall` is the sole classifying signal; `TrackTemp` and
`Compound` corroborate and are recorded but never override. Laps within
{TRANSITION_WINDOW} weather samples of a `Rainfall` state change are classed
`mixed`.

## Files

| File | Contents |
|---|---|
| `manifest.csv` | provenance, versions, cache fingerprint, stage timings |
| `attrition.csv` | CONSORT-style exclusion cascade, row count at each rule |
| `weather_classification.csv` | wet/mixed/dry counts by scope, plus Rainfall-vs-compound agreement |
| `clip_test.csv` | raw (unclipped) vs clipped slope for all five compounds |
| `estimates.csv` | offsets against three baselines and six slope fits, with CIs and diagnostics |
| `stint_summary.csv` | per-`(Driver, Stint)` aggregates for the in-scope compound |
| `robustness.csv` | checks R1-R7, including the R7 falsification test on SOFT/MEDIUM/HARD |
| `thresholds.csv` | the P9 gate, criterion by criterion |

No record-level CSV is written, so no new `.gitignore` pattern is required.

## Interpretation guardrail

These are observational estimates from a single mixed-condition race. Tyre age
in a drying race is also a proxy for elapsed time, so a degradation slope
measured here is confounded with track drying and with fuel burn-off, which
acts in the opposite direction. Where the three fits disagree in sign or the
within-stint interval spans zero, the correct conclusion is that no defensible
parameter exists — not that one of the fits should be preferred.
"""


if __name__ == "__main__":
    raise SystemExit(main())
