# Learnings & Gotchas

Real issues I ran into while building this project — useful for interview prep and future reference.

---

## Data Pipeline

### Unnamed columns from CSV round-tripping
When saving a DataFrame with `to_csv()`, pandas writes the row index as the first column by default. On reload, that index becomes `Unnamed: 0` since it had no header. Fix: use `to_csv(index=False)` when saving, or drop the column on load. A trailing comma in each CSV row can also create a phantom empty column at the end (e.g. `Unnamed: 109`).

### FlightDate loads as a string, not a date
`read_csv` loads date columns as plain strings (`object` dtype). You need `pd.to_datetime()` or `parse_dates=` to convert them — without this, chronological comparisons and temporal splitting won't work.

---

## Feature Engineering & Leakage

### Post-flight columns must be excluded before modelling
The raw dataset has ~110 columns, but most (actual arrival time, taxi minutes, delay-cause breakdowns, cancellation/diversion flags) only exist after the flight lands. A user comparing flights before booking doesn't have access to any of these. Training on them would be data leakage — the model would learn from information unavailable at prediction time, producing misleading accuracy numbers. Identified and documented this during the feature audit, before any modelling code was written.