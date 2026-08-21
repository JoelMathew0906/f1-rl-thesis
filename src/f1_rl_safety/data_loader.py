import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/cache")

def init_fastf1():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_silverstone_race(year: int = 2025):
    """
    Load British GP race session at Silverstone using FastF1.
    Adjust event name if necessary (e.g. 'British Grand Prix', 'Silverstone').
    """
    init_fastf1()
    # You may need to check FastF1 docs/notebooks for exact name/id
    session = fastf1.get_session(year, "Silverstone", "R")
    session.load()
    return session


def extract_stint_and_lap_data(year: int = 2025) -> pd.DataFrame:
    """
    Returns a driver-lap-level DataFrame with lap time, tyre compound,
    tyre age, pit info etc. Use this to calibrate your simulation model.
    """
    session = load_silverstone_race(year)
    laps = session.laps.copy()
    # Keep only relevant columns to start
    keep_cols = [
        "Driver", "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "FreshTyre", "Stint", "TrackStatus", "IsAccurate"
    ]
    laps = laps[keep_cols]
    laps = laps[laps["IsAccurate"] == True]
    return laps


if __name__ == "__main__":
    df = extract_stint_and_lap_data(2025)
    print(df.head())
    df.to_csv("data/silverstone_2025_laps.csv", index=False)