# Likely interview questions and answers

Answers in English (the interview language), with a short 中文提示 for each.

## About the project

**1. Why did you convert to local time instead of staying in UTC?**
Internally UTC is the safest format, and I keep the UTC column. But the German market
trades delivery days and products in local time, so for aggregating per day and for
matching with price data I need Europe/Berlin. Using the timezone database instead of
"+1 hour" means DST days correctly get 23 or 25 hours.
> 中文提示：内部存 UTC，展示和按交割日汇总用本地时间；夏令时那天 23/25 小时。

**2. What happens on the DST change day in your daily summary?**
The day simply has 25 (October) or 23 (March) hourly rows. My daily summary compares the
row count with the true length of that local day, so it keeps DST days but drops the
cut-off first and last days of the forecast window.
> 中文提示：代码里用 `pd.DateOffset(days=1)` 算出每天真实小时数。

**3. Why 120 m wind and not 10 m?**
Modern turbines have hub heights around 100–150 m, where the wind is much stronger and
steadier. Because power scales with v³, using 10 m wind would massively underestimate output.
If 120 m data is missing, I extrapolate from 10 m with the power law (exponent 1/7).
> 中文提示：功率和风速三次方成正比，高度很关键。

**4. How realistic is your power curve?**
It is a stylised curve with typical values (cut-in 3, rated 12, cut-out 25 m/s). Real curves
come from manufacturers and are flatter near rated speed. For a fleet, you'd also account
for wake losses, availability and curtailment. It is good enough to show the shape of a week.
> 中文提示：承认简化，说清楚改进方向。

**5. How would you go from one city to all of Germany?**
Pull forecasts for a grid of points, weight each by installed wind/PV capacity in that region
(e.g. from the Marktstammdatenregister), and sum. Then calibrate against actual generation
from SMARD or ENTSO-E.
> 中文提示：多点 + 装机容量加权 + 用实际发电数据校准。

**6. How would you use this to forecast prices?**
Start simple: regress day-ahead prices on forecast wind, solar and load (residual load =
load − wind − solar is the key driver). Then try gradient boosting with hour-of-day and
weekday features. Always compare to a naive baseline such as "same hour last week".
> 中文提示：关键词 residual load（剩余负荷），以及要有 baseline。

**7. How would a battery use this?**
Charge in hours with high renewables and low prices, discharge in calm evening peaks.
With a price forecast you can solve a small optimisation (e.g. linear programming) over
the next day, subject to capacity, power limits and efficiency.
> 中文提示：低买高卖 + 线性规划 + 约束条件。

**8. How do you know the forecast is good?**
I don't validate it in this version. Next step is to store each day's forecast and compare
with actuals, measuring MAE per lead time. Forecast error also grows with horizon, so day 1
is much more reliable than day 7.
> 中文提示：诚实说没做验证，说出怎么做。

**9. What if the API is down during trading?**
The script already falls back to the last saved response. In production I'd add retries,
logging, alerts, and a second weather provider.
> 中文提示：已有 fallback；生产环境要重试、监控、备用数据源。

**10. What was the hardest part / what did you learn?**
Getting timestamps right: understanding that the API's Unix time is UTC, and that daily
aggregation must happen in local time while being DST-aware. Also realising that the
first and last days of the window are incomplete in local time.
> 中文提示：时间戳和夏令时是最容易出错的地方，说出你踩过的坑。

## Motivation (short prompts)

* **Why Statkraft?** Europe's largest renewable generator, active in trading and flexibility
  (batteries, hydro). The internship sits exactly where data meets the energy transition.
* **Why this team?** Algorithmic trading and asset optimisation turns forecasts into decisions;
  I want to build models that are actually used.
* **What do you bring?** Python and data skills, curiosity about power markets, and the habit of
  building small end-to-end tools quickly (this project is an example).

## Good questions to ask them

* Which data sources and forecast providers does the team rely on most for intraday trading?
* How are models taken from research into production in the team?
* What would a typical intern project look like, and what tools and stack would I use?
