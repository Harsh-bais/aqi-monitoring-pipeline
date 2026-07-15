"""
Step 1: Load raw AQI CSV and reshape it.

The raw CSV has ONE ROW PER (station, pollutant) — e.g. a single station
appears 3-4 times, once for each pollutant it measures (PM2.5, CO, NO2...).

For our pipeline, we want ONE ROW PER STATION with each pollutant as its
own column. This is a classic "long to wide" reshape — pandas calls it
a pivot.
"""

import pandas as pd

RAW_CSV_PATH = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV as-is, no transformation yet."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.station.nunique()} unique stations")
    return df


def pivot_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from long (one row per station+pollutant) to wide
    (one row per station, pollutants as columns).

    We use pollutant_avg as the value — this is what CPCB itself uses
    for AQI sub-index calculation, not min/max.
    """
    # Columns that uniquely identify a station reading
    station_keys = ["country", "state", "city", "station", "last_update", "latitude", "longitude"]

    wide = df.pivot_table(
        index=station_keys,
        columns="pollutant_id",
        values="pollutant_avg",
        aggfunc="first",  # each station+pollutant should only have one row anyway
    ).reset_index()

    # pivot_table adds a confusing column name attribute — clean it up
    wide.columns.name = None

    print(f"Pivoted to {len(wide)} rows (one per station)")
    print(f"Pollutant columns: {[c for c in wide.columns if c not in station_keys]}")
    return wide
RAW_CSV_PATH = r"C:\Users\Harsh Bais\Downloads\3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv"


if __name__ == "__main__":
    raw = load_raw(RAW_CSV_PATH)
    wide = pivot_pollutants(raw)

    print("\nFirst 5 rows of pivoted data:")
    print(wide.head())

    print("\nMissing values per pollutant column:")
    pollutant_cols = [c for c in wide.columns if c not in
                       ["country", "state", "city", "station", "last_update", "latitude", "longitude"]]
    print(wide[pollutant_cols].isna().sum())

    wide.to_csv("aqi_wide.csv", index=False)
    print("\nSaved to aqi_wide.csv")