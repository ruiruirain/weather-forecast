"""
Same price model, two cities: which city's weather forecast explains the German price better?

The model and the split are imported from munich_electrical_price_model.py (not copied),
so both cities use exactly the same formula, the same 17 fit days and the same 4
validation days. Only the city of the weather forecast changes; the price is the same
German day-ahead price in both files.

Data (made by pull_data_and_visualization.py):
    output/munich_2026-10-26_2026-11-15.csv        python pull_data_and_visualization.py
    output/duesseldorf_2026-10-26_2026-11-15.csv   python pull_data_and_visualization.py --city duesseldorf

Steps:
    1. For each city: read the CSV, split by whole days, fit on the first part,
       predict the validation days (each city gets its own fit).
    2. Print the weather coefficients and p-values of both cities side by side.
    3. Print MAE, RMSE and R^2 for the fit and validation days of both cities.
    4. Plot the actual price and both predictions for the validation days.

Usage:
    python compare_munich_and_duesseldorf.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from munich_electrical_price_model import (
    DATA_START, DATA_END, FIT_SHARE, OUTPUT_DIR,
    load_data, split_by_days, day_range, fit_model, scores,
)

CITIES = ["munich", "duesseldorf"]
WEATHER = ["wind_speed_120m", "shortwave_radiation", "temperature_2m"]   # coefficients to compare
PLOT_PATH = OUTPUT_DIR / f"compare_munich_and_duesseldorf_{DATA_START}_{DATA_END}.png"


def csv_path(city: str) -> Path:
    """e.g. output/duesseldorf_2026-10-26_2026-11-15.csv"""
    return OUTPUT_DIR / f"{city}_{DATA_START}_{DATA_END}.csv"


# --------------------------------------------------------------------------
# 1. Fit and validate one city (same steps as munich_electrical_price_model.py)
# --------------------------------------------------------------------------
def fit_and_validate(city: str) -> dict:
    """Fit the model on this city's weather and predict its validation days."""
    path = csv_path(city)
    if not path.exists():
        raise SystemExit(f"{path} not found, run: python pull_data_and_visualization.py --city {city}")
    fit, validate = split_by_days(load_data(path), FIT_SHARE)
    model = fit_model(fit)
    return {"fit": fit, "validate": validate, "model": model, "predicted": model.predict(validate)}


# --------------------------------------------------------------------------
# 2. Coefficients side by side
# --------------------------------------------------------------------------
def coefficient_table(results: dict) -> pd.DataFrame:
    """Weather coefficients (EUR/MWh per +1 m/s, +1 W/m2, +1 degC) and p-values per city."""
    columns = {}
    for city, r in results.items():
        columns[(city, "coef")] = r["model"].params[WEATHER]
        columns[(city, "p-value")] = r["model"].pvalues[WEATHER]
    return pd.DataFrame(columns)


# --------------------------------------------------------------------------
# 3. Errors
# --------------------------------------------------------------------------
def error_table(results: dict) -> pd.DataFrame:
    """MAE, RMSE (EUR/MWh) and R^2 on the fit days and on the validation days, per city."""
    rows = {}
    for city, r in results.items():
        rows[f"{city} fit"] = scores(r["fit"]["price"], r["model"].fittedvalues)
        rows[f"{city} validate"] = scores(r["validate"]["price"], r["predicted"])
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
def plot_comparison(results: dict, errors: pd.DataFrame, path: Path) -> None:
    """Actual price and the prediction of each city's model on the validation days."""
    validate = results[CITIES[0]]["validate"]   # same days and same German price for every city
    t = validate.index.tz_localize(None)        # drop the tz label so matplotlib shows Berlin clock time
    days = validate.index.normalize().unique()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, validate["price"], color="black", linewidth=1.2, label="Actual price")
    for city, r in results.items():
        e = errors.loc[f"{city} validate"]
        ax.plot(t, r["predicted"], linewidth=1,
                label=f"Model with {city.title()} weather "
                      f"(MAE {e['MAE']:.1f}, RMSE {e['RMSE']:.1f} EUR/MWh, R² {e['R2']:.2f})")
    ax.set_ylabel("Day-ahead price\nGermany (EUR/MWh)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())               # one tick per day
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m."))
    ax.set_xlabel("Local time (Europe/Berlin), 15-minute resolution")

    ax.set_title(f"Validation days {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}: "
                 f"same model, weather from {' or '.join(c.title() for c in CITIES)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    results = {city: fit_and_validate(city) for city in CITIES}

    # A fair comparison needs exactly the same validation quarter-hours for every city
    first = results[CITIES[0]]
    for city in CITIES[1:]:
        if not results[city]["validate"].index.equals(first["validate"].index):
            raise SystemExit("The cities have different validation days; "
                             "pull both with the same --start and --end")
    print(f"Fit:      {day_range(first['fit'])}")
    print(f"Validate: {day_range(first['validate'])}")

    print("\nWeather coefficients (EUR/MWh per +1 m/s, +1 W/m2, +1 degC):")
    print(coefficient_table(results).round(4).to_string())

    errors = error_table(results)
    print("\nErrors (MAE and RMSE in EUR/MWh):")
    print(errors.round(2).to_string())

    plot_comparison(results, errors, PLOT_PATH)
    print(f"\nSaved {PLOT_PATH.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
