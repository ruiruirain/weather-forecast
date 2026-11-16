"""
Weather forecast for today and tomorrow in German local time, every 15 minutes.

Why today + tomorrow? Short-term power trading is about the intraday market
(rest of today) and the day-ahead auction (tomorrow). Both trade 15-minute
products, so the weather data uses the same 15-minute resolution.

Pipeline (everything stays in UTC until the last step):
    1. Fetch 15-minute data from Open-Meteo (free, no key) in UTC, covering
       yesterday to tomorrow so that the Berlin days are fully included.
    2. Build a DataFrame indexed by UTC time.
    3. Last step: convert UTC -> Berlin time (DST-safe) and keep only the
       complete local days, which are today and tomorrow.
    4. Plot temperature, wind speed and solar radiation, and save a CSV.

Usage:
    python weather_forecast.py                    # Duesseldorf, live API
    python weather_forecast.py --city husum       # windy North Sea coast
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
LOCAL_TZ = "Europe/Berlin"

LOCATIONS: dict[str, tuple[float, float]] = {  # name: (latitude, longitude)
    "duesseldorf": (51.2277, 6.7735),
    "husum": (54.4858, 9.0524),  # windy coast in Schleswig-Holstein
    "berlin": (52.5200, 13.4050),
    "munich": (48.1374, 11.5755),
}

VARIABLES = [
    "temperature_2m",       # degC
    "wind_speed_10m",       # m/s, standard measuring height
    "wind_speed_120m",      # m/s, roughly wind turbine hub height
    "shortwave_radiation",  # W/m2, global horizontal irradiance (GHI)
]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


# --------------------------------------------------------------------------
# 1. Fetch (UTC)
# --------------------------------------------------------------------------
def fetch_forecast(lat: float, lon: float) -> dict:
    """Call Open-Meteo for 15-minute data, in UTC.

    The Berlin day starts at 23:00 UTC the evening before (22:00 in summer),
    so we also download yesterday (UTC). That way today and tomorrow in Berlin
    time are always complete; the extra hours are cut off in the last step.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ",".join(VARIABLES),
        "past_days": 1,             # yesterday (UTC)
        "forecast_days": 2,         # today + tomorrow (UTC)
        "timezone": "GMT",          # keep everything in UTC; we convert at the end
        "timeformat": "unixtime",   # timestamps as seconds since 1970-01-01 UTC
        "wind_speed_unit": "ms",    # m/s instead of the default km/h
    }
    response = requests.get(API_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# 2. DataFrame in UTC
# --------------------------------------------------------------------------
def to_dataframe(payload: dict) -> pd.DataFrame:
    """Turn the 15-minute block into a DataFrame indexed by UTC time."""
    df = pd.DataFrame(payload["minutely_15"])
    # Unix seconds -> timezone-aware UTC timestamps
    df["time_utc"] = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return df.set_index("time_utc")


# --------------------------------------------------------------------------
# 3. Last step: UTC -> Europe/Berlin, keep today + tomorrow
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


def keep_full_days(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only today and tomorrow in Berlin time."""
    day = df.index.normalize()                       # local midnight of each row
    first = pd.Timestamp.now(tz=LOCAL_TZ).normalize()  # today 00:00
    last = first + pd.DateOffset(days=1)               # tomorrow 00:00
    return df[(day >= first) & (day <= last)]


def format_local(index: pd.DatetimeIndex) -> pd.Index:
    """Readable local time with the zone name, e.g. '2026-11-16 01:00 CET'.

    %Z prints CET in winter and CEST in summer. Only used for display;
    calculations keep the timezone-aware timestamps.
    """
    return index.strftime("%Y-%m-%d %H:%M %Z")


def slots_per_day(df: pd.DataFrame) -> pd.Series:
    """15-minute slots per local day: 96 normally, 92 or 100 on DST days."""
    return df.groupby(df.index.date).size()


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888780"


def plot_forecast(df: pd.DataFrame, city: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = df.index
    midnight = t[0].normalize() + pd.DateOffset(days=1)  # start of tomorrow

    axes[0].plot(t, df["temperature_2m"], color=BLUE, lw=2)
    axes[0].set_ylabel("Temperature\n(°C)")

    axes[1].plot(t, df["wind_speed_120m"], color=BLUE, lw=2, label="120 m (hub height)")
    axes[1].plot(t, df["wind_speed_10m"], color=ORANGE, lw=2, label="10 m")
    axes[1].set_ylabel("Wind speed\n(m/s)")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)

    axes[2].plot(t, df["shortwave_radiation"], color=ORANGE, lw=2)
    axes[2].set_ylabel("Solar radiation\n(W/m²)")

    for ax in axes:
        ax.axvline(midnight, color=GREY, lw=1, ls="--")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    # Label the two market horizons above the top panel
    top = axes[0].get_xaxis_transform()
    axes[0].text(t[0], 1.02, "Today (intraday)", transform=top, color=GREY)
    axes[0].text(midnight, 1.02, "  Tomorrow (day-ahead)", transform=top, color=GREY)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=t.tz))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M", tz=t.tz))
    axes[-1].set_xlabel(f"Local time ({LOCAL_TZ}, CET/CEST), 15-minute resolution")

    fig.suptitle(f"Weather forecast for {city.title()}: today and tomorrow",
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
    args = parser.parse_args()

    lat, lon = LOCATIONS[args.city]
    try:
        payload = fetch_forecast(lat, lon)
    except requests.RequestException as err:
        # Live data only: stop with a short message instead of a long traceback.
        sys.exit(f"Could not reach Open-Meteo ({type(err).__name__}). Check the connection and try again.")
    print("Fetched forecast from Open-Meteo")

    df = to_dataframe(payload)   # everything in UTC
    n_downloaded = len(df)
    df = to_local(df)            # last step: UTC -> Berlin time
    df = keep_full_days(df)      # today + tomorrow as complete Berlin days
    print(f"Downloaded {n_downloaded} quarter-hours (UTC), "
          f"kept {len(df)} (today + tomorrow in Berlin time)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"forecast_{args.city}.csv")
    plot_forecast(df, args.city, OUTPUT_DIR / f"forecast_{args.city}.png")

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    preview = df[["time_utc", *VARIABLES]].head(4).copy()
    preview.index = format_local(preview.index)
    preview["time_utc"] = preview["time_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")
    print("\nFirst rows (UTC vs local time):")
    print(preview)
    print("\n15-minute slots per local day:")
    print(slots_per_day(df).to_string())
    print(f"\nPlot and CSV saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
