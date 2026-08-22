# Learnings & Gotchas

Real issues I ran into while building this project — useful for interview prep and future reference.

---

## Problem Framing

### This is regression on tabular rows, not time-series forecasting
Every row carries a `FlightDate`, which makes the time-series toolchain (`statsmodels`, `prophet`, `sktime`) look relevant. It isn't. Those libraries model **autoregression** — predicting the next value of a series from earlier values of *that same series* (lags, trend, autocorrelation). The test to apply: *does making a prediction require knowing recent observed values of the thing being predicted?* Here it doesn't. The app receives a description of one flight (carrier, flight number, destination, scheduled departure, day of week, month) and predicts that row's `ArrDelay` from its own attributes. There is no series to project forward — and no way to obtain "last Tuesday's actual delay for this flight" at prediction time anyway, since the user is booking months ahead against a static pickled model.

Structurally the data is a panel, not a series: 731 dates carrying 242–555 rows each, where a time series has exactly one observation per time step. `FlightDate` is an attribute of a row, not an index along which the target evolves.

*Reasoned judgement, not a doc citation* — this is a framing decision about the problem, not a claim about how a library behaves.

The consequence cuts both ways, and the second half is the easy one to miss. Ruling out forecasting rules those libraries **out of the stack**. It does **not** rule out time-aware evaluation. Training data is 2024–2025 and users will book 2026, so the honest question is whether a model fit on the past generalises to the future. Rows near each other in time are also genuinely correlated here — a snowstorm or an ATC ground stop hits every flight that day — which is the correlation the docs warn about, even though no autoregressive *model* is involved.

Per the [Cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html) (see `DOCUMENTATION_SOURCES.md`):
> "Time series data is characterized by the correlation between observations that are near in time (autocorrelation). However, classical cross-validation techniques such as `KFold` and `ShuffleSplit` assume the samples are independent and identically distributed, and would result in unreasonable correlation between training and testing instances (yielding poor estimates of generalization error) on time series data."

So: time-awareness enters this project through the **evaluation protocol**, not the model form. The model is a plain regressor; the splitting is time-respecting. The common beginner error is the exact reverse — reaching for a forecasting model and then evaluating it with a shuffled split.

### `ArrDelay` is right-skewed, so the mean describes a flight that rarely happens
Measured on the 324,490 clean rows: mean **4.91**, median **−4**, p75 **11**, std **43.58**, max **3359** (56 hours). 57.2% of SEA departures arrive *early*; 59.2% arrive early or exactly on time. When the mean sits ~9 minutes above the median, a long right tail is dragging the average up while most of the mass sits below it.

Three separate decisions fall out of this one shape:

- **The naive baseline uses the median, not the mean.** The median is the point estimate that minimises absolute error, so it is the correct partner for an MAE-scored project rather than an arbitrary choice. sklearn pairs them the same way — per the [`RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) (see `DOCUMENTATION_SOURCES.md`), `criterion` offers `"absolute_error"` for
  > "the mean absolute error, which minimizes the L1 loss using the median of each terminal node"
- **Training loss and reporting metric disagree.** `LinearRegression` minimises *squared* error while the project reports MAE. Squaring means a single 300-minute miss contributes as much to the objective (90,000) as one hundred 30-minute misses combined (100 × 900). With a max of 3,359 the fit will chase a handful of catastrophic delays that MAE barely notices.
- **What the app may honestly claim.** "Typically lands 4 minutes early" is defensible. "Average delay 4.9 minutes" is arithmetically true and quietly misleading, because the average describes a flight that essentially never occurs.

*The distribution figures are verified against the data; the MAE-versus-RMSE reasoning is a deliberate tradeoff, not a doc-backed rule* — sklearn's metrics page carries no guidance on metric choice under heavy tails.

---

## Data Pipeline

### Unnamed columns from CSV round-tripping
When saving a DataFrame with `to_csv()`, pandas writes the row index as the first column by default. On reload, that index becomes `Unnamed: 0` since it had no header. Fix: use `to_csv(index=False)` when saving, or drop the column on load. A trailing comma in each CSV row can also create a phantom empty column at the end (e.g. `Unnamed: 109`).

### FlightDate loads as a string, not a date
`read_csv` loads date columns as plain strings (`object` dtype). You need `pd.to_datetime()` or `parse_dates=` to convert them — without this, chronological comparisons and temporal splitting won't work.

### A null `ArrDelay` means the flight never arrived — dropping those rows silently scopes the model
Verified across all 24 raw monthly files: of 328,559 SEA-origin rows, 4,069 have a null `ArrDelay`, and **every one is accounted for** — 3,161 cancelled, 908 diverted, zero unexplained. So the nulls are not random missingness to be imputed; they are flights that had no arrival at all.

Dropping them (leaving 324,490) is the right call, since there is no arrival time to regress on. But it defines what the model *is*: **a predictor of arrival delay conditional on the flight operating.** It says nothing about the probability of cancellation or diversion — which for a traveler comparing two flights is often the more important risk. That boundary has to be stated in the app's disclaimer text, not left implicit in a dropped-rows line.

---

## Evaluation & Splitting

### A row-count split doesn't give you the time split you think it does
Flights/day varies 242–555 across the year, with summer running ~34% above winter (Jun–Aug daily mean 513 vs Dec–Feb 383; peak-to-trough month is +43%, Jul 519 vs Jan 361), so the data is not evenly spread across the calendar. That means a fixed *row-count* split (e.g. "first 60% of rows") does not land on a clean date boundary and does not actually cover 60% of the two-year span — it could cover meaningfully more or less time, depending on which season those rows fall in. Fix: split on the **array of unique dates**, not the rows. Decide the split as a percentage of days, find the date boundary, then let row counts per partition follow from that boundary rather than the other way around.

Per the [`TimeSeriesSplit` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) (see `DOCUMENTATION_SOURCES.md`):
> "To ensure comparable metrics across folds, samples must be equally spaced. Once this condition is met, each test set covers the same time duration, while the train set size accumulates data from previous splits."

Verified against this dataset (324,490 rows, 731 dates, `TimeSeriesSplit(n_splits=5)`, sklearn 1.7.2): splitting on raw rows put all 5 fold boundaries mid-date (splitting a single calendar date across train and validation) and produced validation folds spanning 108/125/136/106/124 distinct days at identical row counts. Splitting the date array instead gave exactly equal folds with zero date overlap — dates are evenly spaced (one per calendar day, no gaps across 2024-01-01 to 2025-12-31) even though the flight rows aren't. In the project's actual setup the splitter runs over the **639-date train pool** (2024-01-01 → 2025-09-30, before the 92-date Q4-2025 test partition), giving 106 validation days per fold (639 // 6) with train windows expanding 109 → 533 days.



## Feature Engineering & Leakage

### Post-flight columns must be excluded before modelling
The raw dataset has ~110 columns, but most (actual arrival time, taxi minutes, delay-cause breakdowns, cancellation/diversion flags) only exist after the flight lands. A user comparing flights before booking doesn't have access to any of these. Training on them would be data leakage — the model would learn from information unavailable at prediction time, producing misleading accuracy numbers. Identified and documented this during the feature audit, before any modelling code was written.

### High-cardinality categorical columns make naive one-hot encoding wasteful and risk memorization
`Flight_Number_Reporting_Airline` has 2,591 distinct values. One-hot encoding creates one new 0/1 column per distinct category, so encoding this column alongside `Dest` (96 distinct) and the carrier code (11 distinct) produces on the order of 2,698 columns total — and for any single row, only one column per original feature is `1`; the rest are `0`. That's wasted memory at this scale, but the sparsity isn't only a storage problem: with a key this granular (close to identifying a specific flight), the model risks learning "this exact flight number tends to be late" — a memorized lookup — rather than a generalizable pattern like "evening departures run later across many flights."

Per the [`OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) (see `DOCUMENTATION_SOURCES.md`), the built-in mitigation:
> "min_frequency: Specifies the minimum frequency below which a category will be considered infrequent. If int, categories with a smaller cardinality will be considered infrequent... max_categories: Specifies an upper limit to the number of output features for each input feature when considering infrequent categories."

Decision needed, not left silent: either drop `Flight_Number_Reporting_Airline` as a model feature (keep it only as the grouping key for the naive per-profile baseline) or retain it with `min_frequency`/`max_categories` set and the threshold justified.

### HHMM-encoded time columns are not linear numbers
`CRSDepTime`/`CRSArrTime` store times as an integer that looks like a clock reading (e.g. `1435` = 2:35 PM) rather than a real numeric quantity. Fed to a model as a plain number, the step from 10:59 to 11:00 is +41 in the raw integer for one minute of real elapsed time, and 23:59 to 00:01 is a jump of −2358 for two minutes — the encoding is non-linear in time and discontinuous at midnight.

Checked for malformed encodings and found none: in both the clean CSV and the raw monthly files, `CRSDepTime`/`CRSArrTime` range 1–2359 with zero occurrences of `0`, `2400`, or a minute-part ≥ 60. The `2400`-for-midnight quirk *does* appear, but only in `DepTime`/`ArrTime` — post-flight columns already excluded as leakage. So the problem with these two columns is purely the encoding's shape, not dirty values.

Per the [Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) (see `DOCUMENTATION_SOURCES.md`), this hits model families unevenly:
> "Note that the time related features are passed as is, i.e. without processing them. But this is not much of a problem for tree-based models as they can learn a non-monotonic relationship between ordinal input features and the target."

So raw HHMM specifically breaks linear models (which assume a roughly proportional relationship between a feature's raw value and the target) more than tree-based models, which just split on threshold values and don't care about scale. Fix for linear models: convert to minutes-since-midnight, then apply a periodic encoding (sine/cosine, or `SplineTransformer` with `extrapolation="periodic"`) so midnight wraps around correctly instead of producing a huge jump. Tree models can take the raw ordinal minutes-since-midnight form without the periodic step.


