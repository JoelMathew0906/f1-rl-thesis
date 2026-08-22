import fastf1
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


CACHE_DIR = Path("data/cache")
TRACK_SEGMENTS_CSV = Path("data/silverstone_2024_track_segments.csv")


@dataclass
class TrackSegment:
    """Simple representation of a Silverstone track segment.

    segment_type: "straight" or "corner".
    length: approximate arc length contribution in arbitrary units.
    sector: coarse sector index {1,2,3} along the lap.
    corner_number/name: only populated for corner segments.
    approx_radius: crude radius proxy derived from local geometry.
    start_x/start_y/end_x/end_y: 2D coordinates from FastF1 circuit map.
    """

    id: int
    segment_type: str
    length: float
    sector: int
    corner_number: Optional[int] = None
    corner_name: Optional[str] = None
    approx_radius: Optional[float] = None
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None


def _init_fastf1_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def _build_silverstone_track_segments_from_fastf1(year: int = 2024) -> List[TrackSegment]:
    """Build Silverstone track segments using FastF1 circuit info.

    This uses FastF1's manually curated CircuitInfo.corners data to
    derive an ordered list of straights and corners. Geometry is in
    an arbitrary 2D coordinate system but is sufficient for ordering
    and relative distances.
    """

    _init_fastf1_cache()
    session = fastf1.get_session(year, "Silverstone", "R")
    session.load()

    circuit_info = session.get_circuit_info()
    if circuit_info is None or getattr(circuit_info, "corners", None) is None:
        raise RuntimeError("No circuit info available for Silverstone")

    corners = circuit_info.corners.copy()
    # Expected columns: x, y, number, letter and optionally name
    # Sort by corner number then letter to get lap order
    sort_cols = [c for c in ["Number", "Letter"] if c in corners.columns]
    if sort_cols:
        corners = corners.sort_values(sort_cols)

    xs = corners["X"].to_numpy(dtype=float)
    ys = corners["Y"].to_numpy(dtype=float)

    # cumulative distance along successive corner points
    s_coords = [0.0]
    for i in range(1, len(corners)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        s_coords.append(s_coords[-1] + float(np.hypot(dx, dy)))
    total_len = s_coords[-1] if s_coords else 1.0
    corners["s_coord"] = s_coords

    segments: List[TrackSegment] = []
    seg_id = 0

    for idx in range(len(corners)):
        row = corners.iloc[idx]
        prev_row = corners.iloc[idx - 1] if idx > 0 else corners.iloc[-1]

        # straight from previous corner to this one
        raw_len = float(row["s_coord"] - prev_row["s_coord"]) if idx > 0 else float(total_len - corners.iloc[-1]["s_coord"])
        straight_len = max(raw_len, 0.0)
        # map s_coord proportion to sector index 1..3
        sector = int(1 + (3.0 * float(row["s_coord"]) / max(total_len, 1e-6)))
        sector = int(np.clip(sector, 1, 3))

        seg_id += 1
        segments.append(
            TrackSegment(
                id=seg_id,
                segment_type="Straight",
                length=straight_len,
                sector=sector,
                start_x=float(prev_row["X"]),
                start_y=float(prev_row["Y"]),
                end_x=float(row["X"]),
                end_y=float(row["Y"]),
            )
        )

        # corner at this turn (zero-length segment used for crash modelling)
        radius = None
        radius_len = 0.0
        # approximate radius using distance to neighbours
        radius_len += float(np.hypot(row["X"] - prev_row["X"], row["Y"] - prev_row["Y"]))
        next_row = corners.iloc[(idx + 1) % len(corners)]
        radius_len += float(np.hypot(next_row["X"] - row["X"], next_row["Y"] - row["Y"]))
        if radius_len > 0.0:
            radius = radius_len / 2.0

        # corner naming
        name = None
        if "Name" in corners.columns:
            name = row["Name"]
        if not isinstance(name, str) or not name:
            letter = row["Letter"] if "Letter" in corners.columns else ""
            name = f"Turn {int(row['Number'])}{letter}"

        seg_id += 1
        segments.append(
            TrackSegment(
                id=seg_id,
                segment_type="Corner",
                length=0.0,
                sector=sector,
                corner_number=int(row["Number"]),
                corner_name=str(name),
                approx_radius=radius,
                start_x=float(row["X"]),
                start_y=float(row["Y"]),
                end_x=float(row["X"]),
                end_y=float(row["Y"]),
            )
        )

    return segments


def load_or_build_silverstone_segments(year: int = 2024) -> List[TrackSegment]:
    """Load cached Silverstone track segments or build them via FastF1.

    The first call will build and cache `silverstone_2024_track_segments.csv`
    in the data/ directory; subsequent calls load this CSV directly.
    """

    if TRACK_SEGMENTS_CSV.exists():
        df = pd.read_csv(TRACK_SEGMENTS_CSV)
        segments: List[TrackSegment] = []
        for row in df.itertuples(index=False):
            segments.append(
                TrackSegment(
                    id=int(row.id),
                    segment_type=str(row.segment_type),
                    length=float(row.length),
                    sector=int(row.sector),
                    corner_number=int(row.corner_number)
                    if not pd.isna(row.corner_number)
                    else None,
                    corner_name=str(row.corner_name)
                    if isinstance(row.corner_name, str)
                    and row.corner_name != ""
                    and row.corner_name.lower() != "nan"
                    else None,
                    approx_radius=float(row.approx_radius)
                    if not pd.isna(row.approx_radius)
                    else None,
                    start_x=float(row.start_x)
                    if not pd.isna(row.start_x)
                    else None,
                    start_y=float(row.start_y)
                    if not pd.isna(row.start_y)
                    else None,
                    end_x=float(row.end_x)
                    if not pd.isna(row.end_x)
                    else None,
                    end_y=float(row.end_y)
                    if not pd.isna(row.end_y)
                    else None,
                )
            )
        return segments

    segments = _build_silverstone_track_segments_from_fastf1(year=year)

    # Cache to CSV for reproducibility and faster startup
    df = pd.DataFrame(
        [
            {
                "id": s.id,
                "segment_type": s.segment_type,
                "length": s.length,
                "sector": s.sector,
                "corner_number": s.corner_number,
                "corner_name": s.corner_name,
                "approx_radius": s.approx_radius,
                "start_x": s.start_x,
                "start_y": s.start_y,
                "end_x": s.end_x,
                "end_y": s.end_y,
            }
            for s in segments
        ]
    )
    TRACK_SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACK_SEGMENTS_CSV, index=False)

    return segments
