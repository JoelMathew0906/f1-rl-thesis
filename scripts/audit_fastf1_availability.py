"""Read-only FastF1 feasibility audit for the 2024 British Grand Prix.

Phase 03 audit script. Discovers the sessions FastF1 reports for the 2024
British Grand Prix, attempts to load each one against the existing
`data/cache` directory, and records the observed shape of the data groups
relevant to later wet-session calibration (Phase 04) and circuit-geometry
work (Phase 05): lap timing, tyre compounds, stint structure, weather and
track status, circuit information, and telemetry/position data.

This script performs NO writes to any code path used by the simulator. It
only reads FastF1 sessions (which may fetch-and-cache under `data/cache`,
the same cache directory already used by `src/f1_rl_safety/data_loader.py`
and `src/f1_rl_safety/track.py`) and writes curated, small summary CSVs plus
a manifest and README under `outputs/phase-03-fastf1-audit/<label>/`.

Curated outputs (all small, aggregated tables — never raw per-sample dumps):
  - manifest.csv               one row per audited session: load outcome,
                                timing, cache growth, row/column shape per
                                data group actually loaded.
  - field_audit.csv            one row per (session, data group, column):
                                dtype and null count.
  - weather_summary.csv        one row per session: weather/rainfall summary
                                statistics.
  - reproducibility_check.csv  timing of two consecutive warm-cache loads of
                                the Race session.
  - README.md                  narrative summary of what this run found.

Usage:
    .venv_f1/bin/python scripts/audit_fastf1_availability.py --label LABEL
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

YEAR = 2024
EVENT_NAME = "Silverstone"
SESSION_CODES = ["FP1", "FP2", "FP3", "Q", "R"]
CACHE_DIR = Path("data/cache")
LAPS_KEY_COLUMNS = [
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "IsAccurate",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
]


def init_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def null_count(series: pd.Series) -> int:
    try:
        return int(series.isna().sum())
    except Exception:
        return -1


def audit_laps(session, session_code: str, field_rows: list) -> dict:
    laps = session.laps
    summary = {"laps_rows": len(laps), "laps_cols": laps.shape[1]}
    for col in LAPS_KEY_COLUMNS:
        if col in laps.columns:
            field_rows.append(
                {
                    "session_code": session_code,
                    "data_group": "laps",
                    "column": col,
                    "dtype": str(laps[col].dtype),
                    "n_rows": len(laps),
                    "n_null": null_count(laps[col]),
                }
            )
        else:
            field_rows.append(
                {
                    "session_code": session_code,
                    "data_group": "laps",
                    "column": col,
                    "dtype": "ABSENT",
                    "n_rows": len(laps),
                    "n_null": -1,
                }
            )
    if "IsAccurate" in laps.columns:
        summary["laps_is_accurate_true"] = int((laps["IsAccurate"] == True).sum())  # noqa: E712
    if "Compound" in laps.columns:
        summary["laps_compounds_observed"] = ",".join(
            sorted(str(c) for c in laps["Compound"].dropna().unique())
        )
    if "Driver" in laps.columns:
        summary["laps_distinct_drivers"] = int(laps["Driver"].nunique())
    return summary


def audit_weather(session, session_code: str, field_rows: list, weather_rows: list) -> dict:
    weather = session.weather_data
    summary = {"weather_rows": len(weather), "weather_cols": weather.shape[1]}
    for col in weather.columns:
        field_rows.append(
            {
                "session_code": session_code,
                "data_group": "weather_data",
                "column": col,
                "dtype": str(weather[col].dtype),
                "n_rows": len(weather),
                "n_null": null_count(weather[col]),
            }
        )
    row = {"session_code": session_code, "n_samples": len(weather)}
    if "Rainfall" in weather.columns:
        vc = weather["Rainfall"].value_counts(dropna=False).to_dict()
        row["rainfall_value_counts"] = str(vc)
        row["any_rainfall_true"] = bool(weather["Rainfall"].astype(bool).any())
    if "AirTemp" in weather.columns:
        row["air_temp_min"] = float(weather["AirTemp"].min())
        row["air_temp_max"] = float(weather["AirTemp"].max())
    if "TrackTemp" in weather.columns:
        row["track_temp_min"] = float(weather["TrackTemp"].min())
        row["track_temp_max"] = float(weather["TrackTemp"].max())
    if "Humidity" in weather.columns:
        row["humidity_min"] = float(weather["Humidity"].min())
        row["humidity_max"] = float(weather["Humidity"].max())
    weather_rows.append(row)
    return summary


def audit_track_status(session, session_code: str, field_rows: list) -> dict:
    ts = session.track_status
    summary = {"track_status_rows": len(ts)}
    for col in ts.columns:
        field_rows.append(
            {
                "session_code": session_code,
                "data_group": "track_status",
                "column": col,
                "dtype": str(ts[col].dtype),
                "n_rows": len(ts),
                "n_null": null_count(ts[col]),
            }
        )
    if "Status" in ts.columns:
        summary["track_status_codes_observed"] = ",".join(
            sorted(str(s) for s in ts["Status"].dropna().unique())
        )
    return summary


def audit_circuit_info(session, session_code: str, field_rows: list) -> dict:
    summary = {}
    try:
        ci = session.get_circuit_info()
        if ci is None or getattr(ci, "corners", None) is None:
            summary["circuit_info_available"] = False
            return summary
        corners = ci.corners
        summary["circuit_info_available"] = True
        summary["circuit_corners_rows"] = len(corners)
        for col in corners.columns:
            field_rows.append(
                {
                    "session_code": session_code,
                    "data_group": "circuit_info.corners",
                    "column": col,
                    "dtype": str(corners[col].dtype),
                    "n_rows": len(corners),
                    "n_null": null_count(corners[col]),
                }
            )
    except Exception as exc:  # pragma: no cover - defensive, audit must not crash
        summary["circuit_info_available"] = False
        summary["circuit_info_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def audit_telemetry_and_position(session, session_code: str, field_rows: list) -> dict:
    summary = {}
    try:
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()
        summary["telemetry_rows_fastest_lap"] = len(tel)
        summary["telemetry_cols"] = tel.shape[1]
        for col in tel.columns:
            field_rows.append(
                {
                    "session_code": session_code,
                    "data_group": "telemetry(fastest_lap)",
                    "column": col,
                    "dtype": str(tel[col].dtype),
                    "n_rows": len(tel),
                    "n_null": null_count(tel[col]),
                }
            )
        if "SessionTime" in tel.columns and len(tel) > 1:
            deltas = tel["SessionTime"].diff().dropna().dt.total_seconds()
            summary["telemetry_median_sample_interval_s"] = float(deltas.median())
    except Exception as exc:
        summary["telemetry_error"] = f"{type(exc).__name__}: {exc}"

    try:
        pos = session.pos_data
        if pos is not None and len(pos) > 0:
            summary["position_data_available"] = True
            summary["position_data_n_drivers"] = len(pos)
            sample_key = next(iter(pos.keys()))
            sample_df = pos[sample_key]
            summary["position_data_sample_driver"] = str(sample_key)
            summary["position_data_sample_rows"] = len(sample_df)
            for col in sample_df.columns:
                field_rows.append(
                    {
                        "session_code": session_code,
                        "data_group": f"pos_data(driver={sample_key})",
                        "column": col,
                        "dtype": str(sample_df[col].dtype),
                        "n_rows": len(sample_df),
                        "n_null": null_count(sample_df[col]),
                    }
                )
        else:
            summary["position_data_available"] = False
    except Exception as exc:
        summary["position_data_available"] = False
        summary["position_data_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def audit_session(session_code: str, field_rows: list, weather_rows: list) -> dict:
    record = {"session_code": session_code}
    cache_before = dir_size_bytes(CACHE_DIR)
    t0 = time.time()
    try:
        session = fastf1.get_session(YEAR, EVENT_NAME, session_code)
        session.load()
    except Exception as exc:
        record["load_success"] = False
        record["load_seconds"] = round(time.time() - t0, 3)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["cache_growth_bytes"] = dir_size_bytes(CACHE_DIR) - cache_before
        return record

    record["load_success"] = True
    record["load_seconds"] = round(time.time() - t0, 3)
    record["cache_growth_bytes"] = dir_size_bytes(CACHE_DIR) - cache_before
    record["session_name"] = getattr(session, "name", None)
    record["session_date"] = str(getattr(session, "date", None))

    for audit_fn, kwargs in [
        (audit_laps, dict(session=session, session_code=session_code, field_rows=field_rows)),
        (
            audit_weather,
            dict(
                session=session,
                session_code=session_code,
                field_rows=field_rows,
                weather_rows=weather_rows,
            ),
        ),
        (audit_track_status, dict(session=session, session_code=session_code, field_rows=field_rows)),
        (audit_circuit_info, dict(session=session, session_code=session_code, field_rows=field_rows)),
        (
            audit_telemetry_and_position,
            dict(session=session, session_code=session_code, field_rows=field_rows),
        ),
    ]:
        try:
            record.update(audit_fn(**kwargs))
        except Exception as exc:  # pragma: no cover - defensive, one group must not sink the rest
            record[f"{audit_fn.__name__}_error"] = f"{type(exc).__name__}: {exc}"

    return record


def run_reproducibility_check() -> list:
    """Reload the Race session twice against the warm cache; compare shapes/timings."""
    rows = []
    for attempt in (1, 2):
        t0 = time.time()
        session = fastf1.get_session(YEAR, EVENT_NAME, "R")
        session.load()
        elapsed = round(time.time() - t0, 3)
        rows.append(
            {
                "attempt": attempt,
                "elapsed_seconds": elapsed,
                "laps_rows": len(session.laps),
                "weather_rows": len(session.weather_data),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Timestamp/label for the output directory")
    parser.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated subset of session codes to audit (default: all of "
        + ",".join(SESSION_CODES)
        + "). Used for reduced-scope smoke checks.",
    )
    args = parser.parse_args()

    session_codes = (
        [c.strip() for c in args.sessions.split(",")] if args.sessions else SESSION_CODES
    )

    init_cache()

    out_dir = Path("outputs") / "phase-03-fastf1-audit" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] fetching 2024 event schedule for {EVENT_NAME}...")
    schedule = fastf1.get_event_schedule(YEAR)
    event_rows = schedule[schedule["EventName"].str.contains("British", case=False, na=False)]
    event_row = event_rows.iloc[0] if len(event_rows) else None

    field_rows: list = []
    weather_rows: list = []
    manifest_rows: list = []

    for code in session_codes:
        print(f"[audit] auditing session {code}...")
        try:
            record = audit_session(code, field_rows, weather_rows)
        except Exception:
            record = {
                "session_code": code,
                "load_success": False,
                "error": "UNHANDLED: " + traceback.format_exc(limit=3),
            }
        manifest_rows.append(record)

    print("[audit] running reproducibility check (Race, warm cache x2)...")
    repro_rows = run_reproducibility_check()

    manifest_df = pd.DataFrame(manifest_rows)
    field_df = pd.DataFrame(field_rows)
    weather_df = pd.DataFrame(weather_rows)
    repro_df = pd.DataFrame(repro_rows)

    manifest_df.to_csv(out_dir / "manifest.csv", index=False)
    field_df.to_csv(out_dir / "field_audit.csv", index=False)
    weather_df.to_csv(out_dir / "weather_summary.csv", index=False)
    repro_df.to_csv(out_dir / "reproducibility_check.csv", index=False)

    n_success = int(manifest_df["load_success"].sum()) if len(manifest_df) else 0
    readme_lines = [
        "# Phase 03 FastF1 availability audit — run summary",
        "",
        f"Event: {event_row['EventName'] if event_row is not None else 'UNKNOWN'} "
        f"({event_row['EventDate'] if event_row is not None else '?'}), "
        f"{event_row['Location'] if event_row is not None else '?'}",
        f"Sessions attempted: {', '.join(session_codes)}",
        f"Sessions loaded successfully: {n_success}/{len(session_codes)}",
        "",
        "See manifest.csv for per-session load outcome, timing and cache growth.",
        "See field_audit.csv for per-(session, data group, column) dtype and null counts.",
        "See weather_summary.csv for per-session weather/rainfall summary statistics.",
        "See reproducibility_check.csv for two consecutive warm-cache Race reloads.",
        "",
        "This run performed no deletion, invalidation or relocation of the FastF1 cache,",
        "and made no change to any environment, reward, hazard, tyre, training or",
        "evaluator code.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

    print(f"[audit] wrote outputs to {out_dir}")
    print(f"[audit] sessions loaded: {n_success}/{len(session_codes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
