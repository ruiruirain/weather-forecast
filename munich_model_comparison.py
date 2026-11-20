"""
Five models for the German day-ahead price on the Munich weather forecast, compared on the same days.

Data and split are imported from munich_electrical_price_model.py (not copied):
    output/munich_2026-10-26_2026-11-15.csv, first 17 Berlin days -> fit, last 4 days -> validate
    (the model with yesterday's price fits on 16 days, see below).

Models:
    Weather models (inputs: wind_speed_120m, shortwave_radiation, temperature_2m, Berlin hour)
    Linear   OLS, the model of munich_electrical_price_model.py (the baseline)
    GAM      each weather variable as a smooth curve (cubic B-spline, 5 degrees of freedom)
             instead of a straight line, plus the hour dummies
    XGBoost  many small decision trees added up; fixed settings, no tuning
    Time-series model (input: past prices only)
    ARIMA    ARIMA(1,1,1) on the price. Each validation day is forecast with the actual
             prices up to the end of the day before (already known at the day-ahead
             auction). Parameters are estimated on the fit days only.
    Combination (weather + hour + yesterday's price)
    Linear + yesterday's price
             the linear model with one more input: the actual price at the same time the
             day before (already known at the day-ahead auction). Weather and hour give
             the shape, yesterday's price gives the level. Fit on 16 days, because the
             first day (10-26) has no "yesterday".

Output:
    MAE, RMSE, R^2 and bias on the validation days, one table
    output/munich_model_comparison_2026-10-26_2026-11-15.png, one panel per model

Usage:
    python munich_model_comparison.py
"""
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

from munich_electrical_price_model import (
    CSV_PATH, DATA_START, DATA_END, FIT_SHARE, FORMULA, OUTPUT_DIR,
    load_data, split_by_days, day_range, fit_model, scores,
)

WEATHER = ["wind_speed_120m", "shortwave_radiation", "temperature_2m"]

XGB_SETTINGS = {"max_depth": 3, "eta": 0.05, "seed": 0}   # eta = learning rate
XGB_TREES = 300
ARIMA_ORDER = (1, 1, 1)   # (p, d, q): 1 past value, 1 difference, 1 past shock
YESTERDAY_FORMULA = FORMULA + " + price_yesterday"   # the linear model + one more input

PLOT_PATH = OUTPUT_DIR / f"munich_model_comparison_{DATA_START}_{DATA_END}.png"


# --------------------------------------------------------------------------
# 1. Linear (baseline): the model of munich_electrical_price_model.py
# --------------------------------------------------------------------------
def predict_linear(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    return fit_model(fit).predict(validate)


# --------------------------------------------------------------------------
# 2. GAM: smooth curves for the weather
# --------------------------------------------------------------------------
def clip_to_fit_range(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.DataFrame:
    """B-spline curves only exist inside the range seen when fitting.

    A validation value outside it (e.g. -0.6 degC in Munich on 15 Nov, while the
    fit days never went below 0.2 degC) is put on the edge of that range.
    """
    clipped = validate.copy()
    for column in WEATHER:
        clipped[column] = validate[column].clip(fit[column].min(), fit[column].max())
    return clipped


def gam_formula(fit: pd.DataFrame) -> str:
    """bs(x, df=5): a cubic curve with 2 bends (knots) inside, 5 degrees of freedom.

    By default the knots sit at the 1/3 and 2/3 points of all values. Radiation is 0
    at night (about 60% of the rows), so a knot would land on 0 and the curve would
    break down. Its knots are therefore put at the 1/3 and 2/3 points of the daytime
    values (still 5 degrees of freedom).
    """
    daytime = fit.loc[fit["shortwave_radiation"] > 0, "shortwave_radiation"]
    low, high = daytime.quantile([1 / 3, 2 / 3]).round(1)
    return ("price ~ bs(wind_speed_120m, df=5)"
            f" + bs(shortwave_radiation, knots=({low}, {high}), degree=3)"
            " + bs(temperature_2m, df=5) + C(hour)")


def predict_gam(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    model = smf.ols(gam_formula(fit), data=fit).fit()
    return model.predict(clip_to_fit_range(fit, validate))


# --------------------------------------------------------------------------
# 3. XGBoost: gradient-boosted decision trees
# --------------------------------------------------------------------------
def predict_xgboost(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    columns = WEATHER + ["hour"]   # trees can split on the hour directly, no dummies needed
    train = xgb.DMatrix(fit[columns], label=fit["price"])
    model = xgb.train(XGB_SETTINGS, train, num_boost_round=XGB_TREES)
    return pd.Series(model.predict(xgb.DMatrix(validate[columns])), index=validate.index)


# --------------------------------------------------------------------------
# 4. ARIMA(1,1,1) on the price alone: no weather, no hour
# --------------------------------------------------------------------------
def predict_arima(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    """Forecast one validation day at a time, like a trader before each day-ahead auction.

    Before day D the actual prices up to the end of D-1 are known, so after each
    forecast the actual prices of that day are added to the model (append). The
    parameters stay as estimated on the fit days.
    """
    # numpy array: ARIMA just counts quarter-hours, the dates are kept by us
    with warnings.catch_warnings():
        # only about the optimizer's starting values; convergence is checked below
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
        result = ARIMA(fit["price"].to_numpy(), order=ARIMA_ORDER).fit()
    if not result.mle_retvals.get("converged", True):
        print("Warning: ARIMA did not converge, treat its numbers with care")

    forecasts = []
    for _, today in validate.groupby(validate.index.normalize()):
        forecasts.append(pd.Series(result.forecast(steps=len(today)), index=today.index))
        result = result.append(today["price"].to_numpy())   # this day is now known
    return pd.concat(forecasts)


# --------------------------------------------------------------------------
# 5. Linear + yesterday's price: weather and hour give the shape, yesterday the level
# --------------------------------------------------------------------------
def add_price_yesterday(df: pd.DataFrame) -> pd.DataFrame:
    """Add the price 24 hours earlier, i.e. the same quarter-hour of the day before.

    The first day (10-26) has no day before, so it gets NaN there.
    """
    out = df.copy()
    out["price_yesterday"] = df["price"].shift(freq="24h")   # matched on the time labels
    return out


def predict_linear_yesterday(fit: pd.DataFrame, validate: pd.DataFrame) -> pd.Series:
    """Fit on the fit days that have a "yesterday" (16 days), predict the validation days.

    For validation day D, price_yesterday is the actual price of D-1, which is known
    at the day-ahead auction for D, so this uses no information from the future.
    """
    fit_days = fit.dropna(subset=["price_yesterday"])   # drops 10-26
    model = smf.ols(YESTERDAY_FORMULA, data=fit_days).fit()
    coef = model.params["price_yesterday"]
    print(f"\nLinear + yesterday's price, fit on {day_range(fit_days)}")
    print(f"price_yesterday coefficient {coef:.3f} (p-value {model.pvalues['price_yesterday']:.3g}): "
          f"+1 EUR/MWh yesterday -> {coef:+.2f} EUR/MWh today")
    return model.predict(validate)


# --------------------------------------------------------------------------
# 6. Errors and plot
# --------------------------------------------------------------------------
def errors(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE, RMSE, R^2 (from munich_electrical_price_model.py) and bias.

    bias = mean of (predicted - actual): above 0 means the model is too high on average.
    """
    return {**scores(actual, predicted), "bias": (predicted - actual).mean()}


def plot_models(validate: pd.DataFrame, predictions: dict, table: pd.DataFrame) -> None:
    """One panel per model: actual price (black) and the model's prediction."""
    t = validate.index.tz_localize(None)   # drop the tz label so matplotlib shows Berlin clock time
    days = validate.index.normalize().unique()

    colors = [f"C{i}" for i in range(len(predictions))]   # one matplotlib colour per model
    fig, axes = plt.subplots(len(predictions), 1, figsize=(14, 3 * len(predictions)), sharex=True, sharey=True)
    for ax, (name, predicted), color in zip(axes, predictions.items(), colors):
        e = table.loc[name]
        ax.plot(t, validate["price"], color="black", linewidth=1.2, label="Actual price")
        ax.plot(t, predicted, color=color, linewidth=1, label=name)
        ax.set_title(f"{name}: MAE {e['MAE']:.1f}, RMSE {e['RMSE']:.1f}, "
                     f"bias {e['bias']:+.1f} EUR/MWh, R² {e['R2']:.2f}", fontsize=10)
        ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())               # one tick per day
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m."))
    axes[-1].set_xlabel("Local time (Europe/Berlin), 15-minute resolution")
    fig.suptitle(f"Day-ahead price (Germany), validation days {days[0]:%Y-%m-%d} to "
                 f"{days[-1]:%Y-%m-%d}: weather models (Munich), ARIMA and the combination")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found, run pull_data_and_visualization.py first")

    df = add_price_yesterday(load_data(CSV_PATH))   # extra column, only the last model uses it
    fit, validate = split_by_days(df, FIT_SHARE)
    print(f"Fit:      {day_range(fit)}")
    print(f"Validate: {day_range(validate)}")

    predictions = {
        "Linear": predict_linear(fit, validate),
        "GAM": predict_gam(fit, validate),
        "XGBoost": predict_xgboost(fit, validate),
        "ARIMA": predict_arima(fit, validate),
        "Linear + yesterday's price": predict_linear_yesterday(fit, validate),
    }

    table = pd.DataFrame({name: errors(validate["price"], p) for name, p in predictions.items()}).T
    print("\nValidation errors (MAE, RMSE, bias in EUR/MWh; bias > 0 = model too high):")
    print(table.round(2).to_string())

    plot_models(validate, predictions, table)
    print(f"\nSaved {PLOT_PATH.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
