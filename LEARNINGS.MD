# Learnings & Gotchas

Real issues I ran into while building this project — useful for interview prep and future reference.

---

## Data Pipeline

### Unnamed columns from CSV round-tripping
When saving a DataFrame with `to_csv()`, pandas writes the row index as the first column by default. On reload, that index becomes `Unnamed: 0` since it had no header. Fix: use `to_csv(index=False)` when saving, or drop the column on load. A trailing comma in each CSV row can also create a phantom empty column at the end (e.g. `Unnamed: 109`).

### FlightDate loads as a string, not a date
`read_csv` loads date columns as plain strings (`object` dtype). You need `pd.to_datetime()` or `parse_dates=` to convert them — without this, chronological comparisons and temporal splitting won't work.

---

## Evaluation & Splitting

### A row-count split doesn't give you the time split you think it does
Flights/day varies 242–555 across the year, with summer running ~50% above winter, so the data is not evenly spread across the calendar. That means a fixed *row-count* split (e.g. "first 60% of rows") does not land on a clean date boundary and does not actually cover 60% of the two-year span — it could cover meaningfully more or less time, depending on which season those rows fall in. Fix: split on the **array of unique dates**, not the rows. Decide the split as a percentage of days, find the date boundary, then let row counts per partition follow from that boundary rather than the other way around.

Per the [`TimeSeriesSplit` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) (see `DOCUMENTATION_SOURCES.md`):
> "To ensure comparable metrics across folds, samples must be equally spaced. Once this condition is met, each test set covers the same time duration, while the train set size accumulates data from previous splits."

Verified against this dataset: splitting on raw rows put every fold boundary mid-date (splitting a single calendar date across train and validation) and produced folds spanning 106–136 days at identical row counts. Splitting the 731 unique dates instead gave exactly 106 validation days per fold with zero date overlap — dates are evenly spaced (one per day) even though rows aren't.

---

## Feature Engineering & Leakage

### Post-flight columns must be excluded before modelling
The raw dataset has ~110 columns, but most (actual arrival time, taxi minutes, delay-cause breakdowns, cancellation/diversion flags) only exist after the flight lands. A user comparing flights before booking doesn't have access to any of these. Training on them would be data leakage — the model would learn from information unavailable at prediction time, producing misleading accuracy numbers. Identified and documented this during the feature audit, before any modelling code was written.

### High-cardinality categorical columns make naive one-hot encoding wasteful and risk memorization
`Flight_Number_Reporting_Airline` has 2,591 distinct values. One-hot encoding creates one new 0/1 column per distinct category, so encoding this column alongside `Dest` (96 distinct) and the carrier code (11 distinct) produces on the order of 2,698 columns total — and for any single row, only one column per original feature is `1`; the rest are `0`. That's wasted memory at this scale, but the sparsity isn't only a storage problem: with a key this granular (close to identifying a specific flight), the model risks learning "this exact flight number tends to be late" — a memorized lookup — rather than a generalizable pattern like "evening departures run later across many flights."

Per the [`OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) (see `DOCUMENTATION_SOURCES.md`), the built-in mitigation:
> "min_frequency: Specifies the minimum frequency below which a category will be considered infrequent. If int, categories with a smaller cardinality will be considered infrequent... max_categories: Specifies an upper limit to the number of output features for each input feature when considering infrequent categories."

Decision needed, not left silent: either drop `Flight_Number_Reporting_Airline` as a model feature (keep it only as the grouping key for the naive per-profile baseline) or retain it with `min_frequency`/`max_categories` set and the threshold justified.

### HHMM-encoded time columns are not linear numbers
`CRSDepTime`/`CRSArrTime` store times as an integer that looks like a clock reading (e.g. `1435` = 2:35 PM) rather than a real numeric quantity. Fed to a model as a plain number, the step from 10:59 to 11:00 is +41 in the raw integer for one minute of real elapsed time, and 23:59 to 00:01 is a jump of −2358 for two minutes — the encoding is non-linear in time and discontinuous at midnight. Also verified as a data-quality wrinkle worth checking during cleaning: both columns range 1–2359, with some zero values whose minute-part is ≥60 (a malformed-encoding signal), separate from the linearity problem below.

Per the [Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) (see `DOCUMENTATION_SOURCES.md`), this hits model families unevenly:
> "Note that the time related features are passed as is, i.e. without processing them. But this is not much of a problem for tree-based models as they can learn a non-monotonic relationship between ordinal input features and the target."

So raw HHMM specifically breaks linear models (which assume a roughly proportional relationship between a feature's raw value and the target) more than tree-based models, which just split on threshold values and don't care about scale. Fix for linear models: convert to minutes-since-midnight, then apply a periodic encoding (sine/cosine, or `SplineTransformer` with `extrapolation="periodic"`) so midnight wraps around correctly instead of producing a huge jump. Tree models can take the raw ordinal minutes-since-midnight form without the periodic step.