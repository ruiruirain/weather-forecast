# Presentation outline (10 minutes, 6 slides)

Timing guide in brackets. Keep slides light; the live demo is the strongest part.

---

## Slide 1 — Motivation (1 min)

**Title:** From weather forecast to power output

* German power prices are increasingly driven by wind and solar
* Windy/sunny hours → cheap, sometimes negative prices; calm, dark hours → expensive
* Question: *How much wind and solar output can we expect over the next 7 days?*
* Built as a small, end-to-end Python pipeline: data → cleaning → model → visual

**Say:** "Before I can forecast a price or plan a battery, I need to know what the weather
does to supply. This project is that first building block."

---

## Slide 2 — Architecture (1.5 min)

Simple diagram, left to right:

```
Open-Meteo API  →  JSON (Unix time, UTC)  →  pandas DataFrame (Europe/Berlin)
      →  power curve / PV model  →  plot + CSV + daily summary
```

* Python: `requests`, `pandas`, `numpy`, `matplotlib`
* ~200 lines, one script + one notebook
* Offline fallback with a saved response, so it always runs

---

## Slide 3 — Key code: timestamps (2 min)

```python
df["time_utc"]   = pd.to_datetime(df.pop("time"), unit="s", utc=True)
df["time_local"] = df["time_utc"].dt.tz_convert("Europe/Berlin")
```

* API returns Unix seconds in UTC
* Convert with the timezone database, **never** by adding a fixed hour
* On 25 Oct 2026 the local day has 25 hours, on 29 Mar 23 hours
* Matters because the market delivery day is in local time
* Show the DST example output from the notebook

---

## Slide 4 — Key code: wind power curve (1.5 min)

```python
cf = (v**3 - CUT_IN**3) / (RATED_SPEED**3 - CUT_IN**3)
cf = np.clip(cf, 0, 1)
cf[(v < CUT_IN) | (v >= CUT_OUT)] = 0
```

* Use 120 m wind (hub height), not 10 m: it is roughly 1.5–2× stronger
* Cubic because wind power ∝ v³
* Show the power curve chart from the notebook

---

## Slide 5 — Results + live demo (2.5 min)

* Show the 4-panel plot (temperature, wind, irradiance, estimated output)
* Run `python weather_forecast.py --city husum` live, compare with Düsseldorf
* Read the daily summary: point to the windiest and calmest days
* Interpretation: "Windy days → I'd expect low day-ahead prices; the calm days later in the week
  are where a battery would discharge into higher prices."

---

## Slide 6 — Limitations and next steps (1.5 min)

* One location ≠ Germany → weight multiple sites by installed capacity
* Generic power curve, no curtailment or wake effects
* Validate against actual generation (SMARD, ENTSO-E)
* Then: simple regression of EPEX day-ahead price on wind + solar + load
* Long term: feed price forecasts into a battery schedule optimisation (charge low, discharge high)

**Close:** "This is exactly the kind of pipeline I'd like to build on in your team, with
better data and real trading questions."
