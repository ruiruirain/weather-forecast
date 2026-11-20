"""
Linear model of the German day-ahead price on the Munich weather forecast, every 15 minutes.

Data:
    output/munich_2026-10-26_2026-11-15.csv, made by pull_data_and_visualization.py
    (day-ahead price for Germany + Munich weather forecast made 2 days earlier)

Model (OLS, statsmodels):
    price ~ wind_speed_120m + shortwave_radiation + temperature_2m + C(hour)
    hour = Berlin local hour (0-23); the four quarter-hours of one hour share one dummy.

Steps:
    1. Read the CSV and turn time_utc into Berlin time (DST-safe), as in the other scripts.
    2. Split by whole Berlin days, in time order (no shuffling): the first 80% of the
       days are used to fit the model, the last 20% to validate it.
    3. Fit the model on the first part and print its summary.
    4. Predict the last part and print MAE, RMSE and R^2 (for both parts, to compare).
    5. Plot the actual and the predicted price for the validation days.

Usage:
    python munich_electrical_price_model.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.formula.api as smf

LOCAL_TZ = "Europe/Berlin"

CITY = "munich"
DATA_START, DATA_END = "2026-10-26", "2026-11-15"   # period pulled by pull_data_and_visualization.py

FORMULA = "price ~ wind_speed_120m + shortwave_radiation + temperature_2m + C(hour)"
FIT_SHARE = 0.8   # first 80% of the days -> fit, last 20% -> validate

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CSV_PATH = OUTPUT_DIR / f"{CITY}_{DATA_START}_{DATA_END}.csv"
PLOT_PATH = OUTPUT_DIR / f"{CITY}_price_model_{DATA_START}_{DATA_END}.png"


# --------------------------------------------------------------------------
# 1. Read the data (UTC -> Berlin time)
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and index it by Berlin time, with the local hour as a column."""
    df = pd.read_csv(path)
    df["time_local"] = pd.to_datetime(df["time_local"])
    df = df.set_index("time_local")
    df["hour"] = df.index.hour   # Berlin clock hour, 0-23
    return df


# --------------------------------------------------------------------------
# 2. Split by whole days, in time order
# --------------------------------------------------------------------------
def split_by_days(df: pd.DataFrame, fit_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First fit_share of the Berlin days -> fit, the remaining days -> validate.

    No shuffling: neighbouring quarter-hours are almost the same, so mixing them
    would let the validation part "peek" at the fit part.
    """
    row_day = df.index.normalize()   # local midnight of each row
    days = row_day.unique()          # the Berlin days, in time order
    fitting_days_number = round(fit_share * len(days))   # 21 days -> 17 fit, 4 validate
    if not 1 <= fitting_days_number < len(days):
        raise SystemExit(f"Cannot split {len(days)} days with FIT_SHARE = {fit_share}")
    first_validation_day = days[fitting_days_number]
    return df[row_day < first_validation_day], df[row_day >= first_validation_day]


def day_range(df: pd.DataFrame) -> str:
    """e.g. '17 days (2026-10-26 to 2026-11-11), 1632 quarter-hours'"""
    days = df.index.normalize().unique()
    return (f"{len(days)} days ({days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}), "
            f"{len(df)} quarter-hours")


# --------------------------------------------------------------------------
# 3. Fit
# --------------------------------------------------------------------------
def fit_model(fit: pd.DataFrame):
    """OLS on the fit days only; the validation days are not used here."""
    return smf.ols(FORMULA, data=fit).fit()


# --------------------------------------------------------------------------
# 4. Errors
# --------------------------------------------------------------------------
def scores(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE and RMSE in EUR/MWh, and R^2 (1 = perfect, 0 = as good as the mean, < 0 = worse)."""
    error = actual - predicted
    return {
        "MAE": error.abs().mean(),
        "RMSE": (error ** 2).mean() ** 0.5,
        "R2": 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),
    }


# --------------------------------------------------------------------------
# 5. Plot
# --------------------------------------------------------------------------
def plot_validation(validate: pd.DataFrame, predicted: pd.Series, score: dict, path: Path) -> None:
    """Actual vs predicted price on the validation days."""
    t = validate.index.tz_localize(None)   # drop the tz label so matplotlib shows Berlin clock time
    days = validate.index.normalize().unique()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, validate["price"], linewidth=1, label="Actual price")
    ax.plot(t, predicted, linewidth=1, label="Model prediction")
    ax.set_ylabel("Day-ahead price\nGermany (EUR/MWh)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator())               # one tick per day
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m."))
    ax.set_xlabel("Local time (Europe/Berlin), 15-minute resolution")

    ax.set_title(
        f"Validation days {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}: actual vs model price\n"
        f"MAE {score['MAE']:.1f} EUR/MWh, RMSE {score['RMSE']:.1f} EUR/MWh, R² {score['R2']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found, run pull_data_and_visualization.py first")

    df = load_data(CSV_PATH)
    df_fit, df_validate = split_by_days(df, FIT_SHARE)
    print(f"Fit:      {day_range(df_fit)}")
    print(f"Validate: {day_range(df_validate)}")

    # 3. Fit on the first part
    model = fit_model(df_fit)
    print(model.summary())

    # 4. Predict the last part and compare the error
    series_fit_predicted = model.fittedvalues
    series_validate_predicted = model.predict(df_validate)
    assessment_table = pd.DataFrame({
        "fit": scores(df_fit["price"], series_fit_predicted),
        "validate": scores(df_validate["price"], series_validate_predicted),
    }).T
    print("\n Model assessment:")
    print(assessment_table)

    # 5. Plot the validation days
    plot_validation(df_validate, series_validate_predicted, assessment_table.loc["validate"], PLOT_PATH)
    print(f"\nSaved {PLOT_PATH.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
