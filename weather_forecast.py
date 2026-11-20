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
    python weather_forecast.py                   
    python weather_forecast.py --city husum       # windy North Sea coast
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
LOCAL_TZ = "Europe/Berlin"
# Set once when the module is imported (restart the notebook kernel on a new day)
today_local = pd.Timestamp.now(tz=LOCAL_TZ).normalize()  # today 00:00 Berlin
tomorrow_local = today_local + pd.DateOffset(days=1)     # tomorrow 00:00 Berlin

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
        "forecast_days": 3,         # today + tomorrow (UTC), plus one extra day to avoid truncation at 23:45 UTC
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
def to_dataframe(dict_weather: dict) -> pd.DataFrame:
    """Turn the 15-minute block into a DataFrame indexed by UTC time."""
    df = pd.DataFrame(dict_weather["minutely_15"])
    # Unix timestamp -> UTC
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
    return df[(day >= today_local) & (day <= tomorrow_local)]

# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
def plot_forecast(df: pd.DataFrame, city: str, path: Path) -> None:
    """Three panels (temperature, wind, solar); dashed line = start of tomorrow."""
    t = df.index.tz_localize(None)   # drop the tz label so matplotlib shows Berlin clock time

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t, df["temperature_2m"])
    axes[0].set_ylabel("Temperature (°C)")

    axes[1].plot(t, df["wind_speed_120m"], label="120 m (hub height)")
    axes[1].plot(t, df["wind_speed_10m"], label="10 m")
    axes[1].set_ylabel("Wind speed (m/s)")
    axes[1].legend()

    axes[2].plot(t, df["shortwave_radiation"])
    axes[2].set_ylabel("Solar radiation (W/m²)")

    for ax in axes:
        ax.axvline(tomorrow_local.tz_localize(None), color="grey", linestyle="--")

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m. %H:%M"))
    axes[2].set_xlabel("Local time (Europe/Berlin)")

    fig.suptitle(f"Weather forecast for {city.title()}: today and tomorrow")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--city", default="munich", choices=LOCATIONS)
    args = parser.parse_args()

    lat, lon = LOCATIONS[args.city]
    try:
        dict_weather = fetch_forecast(lat, lon)
    except requests.RequestException as err:
        raise SystemExit(f"Could not fetch Open-Meteo ({err})") 
    print("Fetched forecast from Open-Meteo")

    df = to_dataframe(dict_weather)   # everything in UTC
    df = to_local(df)            # last step: UTC -> Berlin time
    df = keep_full_days(df)      # today + tomorrow as complete Berlin days
    print(f"kept {len(df)} data (today + tomorrow in Berlin time)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = f"forecast_{args.city}_{today_local:%Y-%m-%d}"   # e.g. forecast_munich_2026-11-17
    df.to_csv(OUTPUT_DIR / f"{stem}.csv")
    plot_forecast(df, args.city, OUTPUT_DIR / f"{stem}.png")

if __name__ == "__main__":
    main()
