"""
German day-ahead prices and one city's weather forecasts, every 15 minutes, plotted together.

Data (public, no API key):
    prices   Energy-Charts API: day-ahead price for Germany (bidding zone DE-LU),
             Bundesnetzagentur | SMARD.de, CC BY 4.0
    weather  Open-Meteo Previous Runs API, one city (--city, default Munich): the forecast made 2 days before
             each quarter-hour, so it was already known before the day-ahead auction

Pipeline (everything stays in UTC until the last step), same as weather_forecast.py:
    1. Fetch prices and weather in UTC, one extra day on each side, so that the
       Berlin days are fully included.
    2. Build DataFrames indexed by UTC time, align the radiation, join them.
    3. Last step: convert UTC -> Berlin time (DST-safe) and keep only the
       complete local days from --start to --end.
    4. Plot price, temperature, wind speed and solar radiation, and save a CSV.

Usage:
    python pull_data_and_visualization.py                                     # 2026-10-26 to 2026-11-15
    python pull_data_and_visualization.py --start 2026-11-01 --end 2026-11-14
    python pull_data_and_visualization.py --city duesseldorf                  # output/duesseldorf_2026-10-26_2026-11-15
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests

PRICE_URL = "https://api.energy-charts.info/price"
PAST_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LOCAL_TZ = "Europe/Berlin"

LOCATIONS: dict[str, tuple[float, float]] = {  # name: (latitude, longitude), same as weather_forecast.py
    "duesseldorf": (51.2277, 6.7735),
    "husum": (54.4858, 9.0524),  # windy coast in Schleswig-Holstein
    "berlin": (52.5200, 13.4050),
    "munich": (48.1374, 11.5755),
}
DEFAULT_CITY = "munich"

VARIABLES = [
    "temperature_2m",       # degC
    "wind_speed_120m",      # m/s, roughly wind turbine hub height
    "shortwave_radiation",  # W/m2, mean of the preceding 15 minutes
]
LEAD_DAYS = 2   # forecast made 2 days before -> known before the day-ahead auction
MIN_DAYS = 7    # shortest period allowed

DEFAULT_START = "2026-10-26"
DEFAULT_END = "2026-11-15"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


# --------------------------------------------------------------------------
# 1. Fetch (UTC)
# --------------------------------------------------------------------------
def fetch_past_weather(lat: float, lon: float, start: str, end: str) -> dict:
    """15-minute weather forecasts as they looked LEAD_DAYS days earlier, in UTC."""
    suffix = f"_previous_day{LEAD_DAYS}"
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ",".join(v + suffix for v in VARIABLES),
        "start_date": start,        # first UTC day (included)
        "end_date": end,            # last UTC day (included)
        "timezone": "GMT",          # keep everything in UTC; we convert at the end
        "timeformat": "unixtime",   # timestamps as seconds since 1970-01-01 UTC
        "wind_speed_unit": "ms",    # m/s instead of the default km/h
    }
    response = requests.get(PAST_RUNS_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_prices(start: str, end: str) -> dict:
    """Day-ahead prices for Germany (DE-LU), every 15 minutes, in EUR/MWh."""
    params = {"bzn": "DE-LU", "start": start, "end": end}
    response = requests.get(PRICE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# 2. DataFrames in UTC
# --------------------------------------------------------------------------
def weather_to_dataframe(payload: dict) -> pd.DataFrame:
    """Turn the 15-minute weather block into a DataFrame indexed by UTC time."""
    df = pd.DataFrame(payload["minutely_15"])
    # Unix seconds -> timezone-aware UTC timestamps
    df["time_utc"] = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    df = df.set_index("time_utc")
    # "temperature_2m_previous_day2" -> "temperature_2m"
    suffix = f"_previous_day{LEAD_DAYS}"
    return df.rename(columns=lambda name: name.replace(suffix, ""))


def prices_to_dataframe(payload: dict) -> pd.DataFrame:
    """Turn the price response into a DataFrame indexed by UTC time."""
    df = pd.DataFrame({"time": payload["unix_seconds"], "price": payload["price"]})
    # Unix seconds -> timezone-aware UTC timestamps
    df["time_utc"] = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return df.set_index("time_utc")


def align_radiation(weather: pd.DataFrame) -> pd.DataFrame:
    """Make radiation describe the same quarter-hour as the price.

    Open-Meteo's radiation at 12:15 is the mean of 12:00-12:15, while the price
    at 12:00 covers 12:00-12:15. Taking 15 minutes off the radiation time labels
    puts that value on 12:00. Temperature and wind are values at the exact time,
    so they stay as they are.
    """
    radiation = weather["shortwave_radiation"].copy()
    radiation.index = radiation.index - pd.Timedelta("15min")
    aligned = weather.drop(columns="shortwave_radiation")
    aligned["shortwave_radiation"] = radiation   # matched on the new time labels
    return aligned


# --------------------------------------------------------------------------
# 3. Last step: UTC -> Europe/Berlin, keep --start to --end
# --------------------------------------------------------------------------
def to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the UTC index to Berlin time (the only time-zone conversion).

    The tz database knows the CET/CEST switch, so a DST change day correctly
    gets 92 or 100 quarter-hours. Never do "UTC + 1h".
    """
    local = df.copy()
    local["time_utc"] = df.index  # keep UTC as a column for reference
    local.index = df.index.tz_convert(LOCAL_TZ)
    local.index.name = "time_local"
    return local


def keep_full_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep only the Berlin days from start to end (both included)."""
    first_day = pd.Timestamp(start, tz=LOCAL_TZ)   # start 00:00 Berlin
    last_day = pd.Timestamp(end, tz=LOCAL_TZ)      # end 00:00 Berlin
    day = df.index.normalize()                     # local midnight of each row
    return df[(day >= first_day) & (day <= last_day)]


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
def plot_data(df: pd.DataFrame, path: Path, city: str, start: str, end: str) -> None:
    """Four panels on one time axis: price, temperature, wind speed, radiation."""
    t = df.index.tz_localize(None)   # drop the tz label so matplotlib shows Berlin clock time
    name = city.title()              # "munich" -> "Munich"

    panels = [
        ("price", "Day-ahead price\nGermany (EUR/MWh)"),
        ("temperature_2m", f"Temperature\n{name} (°C)"),
        ("wind_speed_120m", f"Wind speed 120 m\n{name} (m/s)"),
        ("shortwave_radiation", f"Solar radiation\n{name} (W/m²)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 10), sharex=True)

    for ax, (column, label) in zip(axes, panels):
        ax.plot(t, df[column], linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].axhline(0, color="grey", linewidth=0.8)   # prices can go below zero

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())           # one tick per day
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    axes[-1].tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("Local time (Europe/Berlin), 15-minute resolution")

    fig.suptitle(f"Day-ahead price (Germany) and weather forecast ({name}), {start} to {end}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--start", default=DEFAULT_START, help="first Berlin day, YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="last Berlin day, YYYY-MM-DD")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=LOCATIONS, help="city for the weather forecast")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end:
        raise SystemExit(f"--start ({args.start}) is after --end ({args.end})")
    n_days = (end - start).days + 1
    if n_days < MIN_DAYS:
        raise SystemExit(f"Need at least {MIN_DAYS} days, got {n_days} ({args.start} to {args.end})")

    # One extra day on each side: the Berlin day starts at 23:00 UTC the evening
    # before (22:00 in summer). The extra hours are cut off in the last step.
    pad_start = (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pad_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    lat, lon = LOCATIONS[args.city]
    try:
        weather_payload = fetch_past_weather(lat, lon, pad_start, pad_end)
        price_payload = fetch_prices(pad_start, pad_end)
    except requests.RequestException as err:
        raise SystemExit(f"Could not fetch data ({err})")
    print(f"Fetched prices from Energy-Charts and {args.city} weather from Open-Meteo")

    weather = align_radiation(weather_to_dataframe(weather_payload))   # UTC
    prices = prices_to_dataframe(price_payload)                         # UTC
    df = prices.join(weather, how="inner")        # everything in UTC
    df = to_local(df)                             # last step: UTC -> Berlin time
    df = keep_full_days(df, args.start, args.end)
    print(f"kept {len(df)} data ({n_days} Berlin days, {args.start} to {args.end})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = f"{args.city}_{args.start}_{args.end}"   # e.g. munich_2026-10-26_2026-11-15
    df.to_csv(OUTPUT_DIR / f"{stem}.csv")
    plot_data(df, OUTPUT_DIR / f"{stem}.png", args.city, args.start, args.end)
    print(f"Saved {stem}.csv and {stem}.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
