"""
Step 2: Calculate AQI per station using CPCB's official sub-index method.

How CPCB's AQI actually works:
- Each pollutant has its own breakpoint table mapping concentration ->
  a 0-500 sub-index.
- The station's overall AQI = the MAX sub-index among available
  pollutants (not an average) - the worst pollutant drives the AQI,
  since that's the one actually harming health.
- CPCB requires at least 3 pollutants (including PM2.5 or PM10) for a
  "valid" AQI. Fewer than that = insufficient data, we flag it rather
  than silently computing a misleading number.

Breakpoint values below are CPCB's official ranges (National Air
Quality Index, 2014), for pollutant concentrations in ug/m3 (mg/m3 for
CO), 24-hr averages for most pollutants.
"""

import pandas as pd

# Each entry: (conc_low, conc_high, aqi_low, aqi_high)
BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500),
    ],
    "CO": [  # note: CO breakpoints are in mg/m3, our data is in the same unit from the API
        (0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
        (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500),
    ],
    "OZONE": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
        (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500),
    ],
}

CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

MIN_POLLUTANTS_REQUIRED = 3
PRIMARY_POLLUTANTS = {"PM2.5", "PM10"}


def sub_index(pollutant: str, concentration: float) -> float | None:
    """Convert a raw pollutant concentration into its CPCB sub-index (0-500)."""
    if pd.isna(concentration) or pollutant not in BREAKPOINTS:
        return None

    for conc_low, conc_high, aqi_low, aqi_high in BREAKPOINTS[pollutant]:
        if conc_low <= concentration <= conc_high:
            # Linear interpolation within the breakpoint band
            return round(
                ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low,
                1,
            )

    # Concentration is above the highest defined breakpoint - cap at 500
    if concentration > BREAKPOINTS[pollutant][-1][1]:
        return 500.0
    return None


def category_for_aqi(aqi: float) -> str:
    for low, high, label in CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Severe"  # anything above 500 is still Severe


def calculate_station_aqi(row: pd.Series, pollutant_cols: list[str]) -> pd.Series:
    """
    Given one station's row (with pollutant concentration columns),
    return its overall AQI, dominant pollutant, category, and whether
    the reading meets CPCB's minimum-pollutant validity rule.
    """
    sub_indices = {}
    for p in pollutant_cols:
        if p in row and pd.notna(row[p]):
            si = sub_index(p, row[p])
            if si is not None:
                sub_indices[p] = si

    available_count = len(sub_indices)
    has_primary = any(p in sub_indices for p in PRIMARY_POLLUTANTS)
    is_valid = available_count >= MIN_POLLUTANTS_REQUIRED and has_primary

    if not sub_indices:
        return pd.Series({
            "aqi": None, "dominant_pollutant": None,
            "category": None, "is_valid": False, "pollutants_used": 0,
        })

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    aqi_value = sub_indices[dominant_pollutant]

    return pd.Series({
        "aqi": aqi_value,
        "dominant_pollutant": dominant_pollutant,
        "category": category_for_aqi(aqi_value),
        "is_valid": is_valid,
        "pollutants_used": available_count,
    })


if __name__ == "__main__":
    df = pd.read_csv("aqi_wide.csv")
    pollutant_cols = ["CO", "NH3", "NO2", "OZONE", "PM10", "PM2.5", "SO2"]

    aqi_results = df.apply(lambda row: calculate_station_aqi(row, pollutant_cols), axis=1)
    df = pd.concat([df, aqi_results], axis=1)

    print(f"Total stations: {len(df)}")
    print(f"Stations with valid AQI (>=3 pollutants incl. PM): {df.is_valid.sum()}")
    print(f"Stations with insufficient data: {(~df.is_valid).sum()}")
    print()
    print("AQI category distribution (valid stations only):")
    print(df[df.is_valid].category.value_counts())
    print()
    print("Sample results:")
    print(df[df.is_valid][["city", "station", "aqi", "dominant_pollutant", "category"]].head(10))
    print()
    print("Most hazardous stations right now:")
    print(df[df.is_valid].nlargest(5, "aqi")[["city", "station", "aqi", "dominant_pollutant", "category"]])

    df.to_csv("aqi_with_scores.csv", index=False)
    print("\nSaved to aqi_with_scores.csv")