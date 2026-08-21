import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/cache")


def init_fastf1():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_silverstone_race(year: int = 2024):
    """Load British GP race session at Silverstone using FastF1."""
    init_fastf1()
    session = fastf1.get_session(year, "Silverstone", "R")
    session.load()
    return session


def extract_stint_and_lap_data(year: int = 2024) -> pd.DataFrame:
    """Return driver-lap-level DataFrame for calibration.

    The resulting CSV is used by F1RaceEnv to calibrate base lap time,
    compound offsets and degradation slopes for the chosen season.
    """
    session = load_silverstone_race(year)
    laps = session.laps.copy()
    keep_cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "Stint",
        "TrackStatus",
        "IsAccurate",
    ]
    laps = laps[keep_cols]
    laps = laps[laps["IsAccurate"] == True]
    return laps


if __name__ == "__main__":
    df = extract_stint_and_lap_data(2024)
    print(df.head())
    df.to_csv("data/silverstone_2024_laps.csv", index=False)
