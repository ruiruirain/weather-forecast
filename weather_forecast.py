"""
7-day weather forecast for a German location -> wind & solar power proxies.

Pipeline:
    1. Fetch an hourly forecast from the Open-Meteo API (free, no API key).
    2. Convert Unix timestamps (UTC) to Europe/Berlin local time, DST-safe.
    3. Estimate wind and solar output with simple physical models.
    4. Plot the series and write a CSV + a daily summary.

Usage:
    python weather_forecast.py                    # Duesseldorf, live API
    python weather_forecast.py --city husum       # windy North Sea coast
    python weather_forecast.py --offline          # use the saved sample response
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
LOCAL_TZ = "Europe/Berlin"
HUB_HEIGHT = 120  # metres; Open-Meteo offers wind speed at 10, 80, 120, 180 m

LOCATIONS = {  # name: (latitude, longitude)
    "duesseldorf": (51.2277, 6.7735),
    "husum": (54.4858, 9.0524),  # onshore wind hub in Schleswig-Holstein
    "berlin": (52.5200, 13.4050),
    "munich": (48.1374, 11.5755),  # more sun, less wind
}

HOURLY_VARS = [
    "temperature_2m",         # degC
    "wind_speed_10m",         # m/s
    f"wind_speed_{HUB_HEIGHT}m",  # m/s, roughly turbine hub height
    "shortwave_radiation",    # W/m2, global horizontal irradiance (GHI)
]

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = BASE_DIR / "data" / "sample_response.json"
OUTPUT_DIR = BASE_DIR / "output"


# --------------------------------------------------------------------------
# 1. Fetch
# --------------------------------------------------------------------------
def fetch_forecast(lat: float, lon: float, days: int = 7) -> dict:
    """Call Open-Meteo and return the raw JSON payload."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "GMT",          # ask for UTC; we convert ourselves
        "timeformat": "unixtime",   # timestamps as seconds since 1970-01-01 UTC
        "wind_speed_unit": "ms",    # m/s instead of the default km/h
    }
    response = requests.get(API_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# 2. Timestamps: Unix seconds (UTC) -> Europe/Berlin
# --------------------------------------------------------------------------
def to_dataframe(payload: dict) -> pd.DataFrame:
    """Turn the 'hourly' block into a DataFrame indexed by local time."""
    df = pd.DataFrame(payload["hourly"])

    # Unix seconds -> timezone-aware UTC timestamps
    df["time_utc"] = pd.to_datetime(df.pop("time"), unit="s", utc=True)

    # UTC -> local time. The tz database knows the CET/CEST switch, so a
    # DST change day correctly gets 23 or 25 hours. Never do "UTC + 1h".
    df["time_local"] = df["time_utc"].dt.tz_convert(LOCAL_TZ)

    return df.set_index("time_local")


# --------------------------------------------------------------------------
# 3. Power proxies
# --------------------------------------------------------------------------
CUT_IN, RATED_SPEED, CUT_OUT = 3.0, 12.0, 25.0  # m/s, generic onshore turbine
PV_PERFORMANCE_RATIO = 0.85  # losses: inverter, temperature, soiling, ...


def fill_hub_wind(df: pd.DataFrame) -> pd.Series:
    """Hub-height wind; if missing, extrapolate from 10 m with the power law."""
    hub_col = f"wind_speed_{HUB_HEIGHT}m"
    alpha = 1 / 7  # wind shear exponent for open land
    extrapolated = df["wind_speed_10m"] * (HUB_HEIGHT / 10) ** alpha
    if hub_col not in df:
        return extrapolated
    return df[hub_col].fillna(extrapolated)


def wind_capacity_factor(wind_speed) -> np.ndarray:
    """Simplified turbine power curve, returns output as a share of capacity.

    0 below cut-in, cubic ramp up to rated speed, 1 until cut-out, then 0
    (the turbine shuts down in storms to protect itself).
    """
    v = np.asarray(wind_speed, dtype=float)
    cf = (v**3 - CUT_IN**3) / (RATED_SPEED**3 - CUT_IN**3)
    cf = np.clip(cf, 0.0, 1.0)
    cf[(v < CUT_IN) | (v >= CUT_OUT)] = 0.0
    return cf


def solar_capacity_factor(ghi) -> np.ndarray:
    """PV output share: irradiance relative to 1000 W/m2 test conditions."""
    return np.clip(np.asarray(ghi, dtype=float) / 1000 * PV_PERFORMANCE_RATIO, 0, 1)


def add_power_estimates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wind_hub"] = fill_hub_wind(df)
    df["wind_cf"] = wind_capacity_factor(df["wind_hub"])
    df["solar_cf"] = solar_capacity_factor(df["shortwave_radiation"])
    return df


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per local calendar day (the delivery day used by the market)."""
    daily = df.resample("D").agg(
        hours=("temperature_2m", "size"),
        temp_mean=("temperature_2m", "mean"),
        wind_hub_mean=("wind_hub", "mean"),
        wind_cf=("wind_cf", "mean"),
        solar_cf=("solar_cf", "mean"),
    )
    # A local day normally has 24 hours, but 23 or 25 on a DST change day.
    # Adding "1 calendar day" in local time gives the true day length.
    day_start = daily.index
    expected_hours = (day_start + pd.DateOffset(days=1) - day_start) / pd.Timedelta(hours=1)
    # The 7-day UTC window cuts off the first/last local day: keep full days only.
    daily = daily[daily["hours"] == expected_hours]
    daily.index = daily.index.date
    return daily.round(2)


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888780"


def plot_forecast(df: pd.DataFrame, city: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = df.index

    axes[0].plot(t, df["temperature_2m"], color=BLUE, lw=2)
    axes[0].set_ylabel("Temperature\n(°C)")

    axes[1].plot(t, df["wind_hub"], color=BLUE, lw=2, label=f"{HUB_HEIGHT} m (hub)")
    axes[1].plot(t, df["wind_speed_10m"], color=ORANGE, lw=2, label="10 m")
    axes[1].axhline(CUT_IN, color=GREY, lw=1, ls="--")
    axes[1].axhline(RATED_SPEED, color=GREY, lw=1, ls="--")
    for y, label in [(RATED_SPEED, "rated"), (CUT_IN, "cut-in")]:
        axes[1].annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                         xytext=(4, 0), textcoords="offset points",
                         va="center", color=GREY, fontsize=8)
    axes[1].set_ylabel("Wind speed\n(m/s)")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)

    axes[2].plot(t, df["shortwave_radiation"], color=ORANGE, lw=2)
    axes[2].set_ylabel("Solar radiation\n(W/m²)")

    axes[3].plot(t, df["wind_cf"] * 100, color=BLUE, lw=2, label="Wind")
    axes[3].plot(t, df["solar_cf"] * 100, color=ORANGE, lw=2, label="Solar PV")
    axes[3].set_ylabel("Est. output\n(% of capacity)")
    axes[3].set_ylim(0, 105)
    axes[3].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)

    for ax in axes:
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz=t.tz))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=t.tz))
    axes[-1].set_xlabel(f"Local time ({LOCAL_TZ})")

    fig.suptitle(f"7-day forecast for {city.title()}: weather and power proxies",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--city", default="duesseldorf", choices=LOCATIONS)
    parser.add_argument("--offline", action="store_true",
                        help="use data/sample_response.json instead of the API")
    args = parser.parse_args()

    if args.offline:
        payload = json.loads(SAMPLE_FILE.read_text())
        print(f"Loaded saved sample: {SAMPLE_FILE.name}")
    else:
        lat, lon = LOCATIONS[args.city]
        try:
            payload = fetch_forecast(lat, lon)
            SAMPLE_FILE.parent.mkdir(exist_ok=True)
            SAMPLE_FILE.write_text(json.dumps(payload))  # backup for offline demos
            print(f"Fetched {len(payload['hourly']['time'])} hours from Open-Meteo")
        except requests.RequestException as err:
            # No internet during a live demo? Fall back to the last saved response.
            print(f"API call failed ({type(err).__name__}); using saved sample instead")
            payload = json.loads(SAMPLE_FILE.read_text())

    df = add_power_estimates(to_dataframe(payload))
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"forecast_{args.city}.csv")
    plot_forecast(df, args.city, OUTPUT_DIR / f"forecast_{args.city}.png")

    print("\nFirst rows (UTC vs local time):")
    print(df[["time_utc", "temperature_2m", "wind_hub", "wind_cf"]].head(3))
    print("\nDaily summary (local calendar days):")
    print(daily_summary(df))
    print(f"\nPlot and CSV saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
