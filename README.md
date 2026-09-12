# Seattle Flight Reliability

## Contents

- [Introduction](#introduction)
- [Steps](#steps)
- [Results](#results)
- [Usage](#usage)
- [Next Steps](#next-steps)
- [Repository Layout](#repository-layout)
- [References](#references)

## Introduction

Compares two Seattle-origin domestic flights on historical arrival delay, using only what a traveler knows before booking: carrier, destination, scheduled times, and date. Built on US DOT on-time records for 2024-2025.

**Status:** the median baseline is built and scored. No candidate model is trained yet.

## Steps

> 1. [Audit the raw data](./notebooks/data_quality_audit.ipynb) — nulls, duplicates, month coverage → [`reports/data_quality_audit.md`](./reports/data_quality_audit.md)
> 2. [Classify all 110 raw columns](./reports/feature_audit.md) as target, pre-booking, post-flight or drop, so no post-flight field leaks into training
>    |Target|Pre-booking|Post-flight|Drop|
>    |-:|-:|-:|-:|
>    |1|**13**|71|25|
> 3. [Filter and clean](./notebooks/data_cleaning.ipynb) → `data/interim/seattle_ontime_clean.csv`
>    |Stage|Rows|
>    |:-|-:|
>    |Raw download|14,080,680|
>    |SEA departures|328,559|
>    |Non-null `ArrDelay`|**324,490**|
>
>    Rows without an `ArrDelay` are cancelled or diverted, so the project is scoped to flights that operate and arrive.
> 4. [Split on dates, not rows](./notebooks/linear_baseline.ipynb) — flight counts run 242-555 per day, so a row split cuts individual days in half
>    |Partition|Period|Flights|Dates|
>    |:-|:-|-:|-:|
>    |Train pool|2024-01-01 to 2025-09-30|285,649|639|
>    |Held-out test|2025-10-01 to 2025-12-31|38,841|92|
>
>    Five `TimeSeriesSplit` folds run over the unique-date array, then map back to rows. `gap=0`, since no feature is lagged or rolling. The test partition is Q4 only, so its score will describe winter operations.
>
>    ![Cross-validation fold structure](./reports/figures/cv_fold_structure.png)
> 5. [Build a 3-level median baseline](./notebooks/linear_baseline.ipynb) — medians match MAE and resist the long right tail. Level 2 was chosen from 11 candidate groupings: carrier + departure hour lifted 0.415 min over the global median, against 0.149 for carrier + destination
>    1. Carrier + flight number + destination, n ≥ 10
>    2. Carrier + scheduled departure hour, n ≥ 10
>    3. Global training median
> 6. [Encode the features](./notebooks/linear_baseline.ipynb) for a linear candidate
>    - `CRSDepTime` and `CRSArrTime` are HHMM clock readings, not quantities. Converted to minutes since midnight, then sin/cos encoded so 23:59 and 00:01 sit 0.0087 apart instead of 1,438
>    - `drop="first"`, since an unregularised `LinearRegression` leaves coefficients undetermined otherwise
>    - Two encoders, split by whether the category set is closed. `Month` and `DayOfWeek` get explicit categories; carrier and destination cannot be enumerated ahead of time
>
>    This surfaced a defect in scikit-learn itself. The library turns text columns like carrier codes into numeric ones, and warns you when it encounters a value it never saw during training. That warning was written from the encoder's settings alone, without checking what the encoder was actually doing, so it could report that a value was handled one way when it was really handled another way. Fixed upstream in [scikit-learn#34861](https://github.com/scikit-learn/scikit-learn/pull/34861), merged September 2026.

## Results

| Metric | Value |
|---|---:|
| Mean cross-validated MAE | **19.95 min** |
| Standard deviation across folds | 1.21 min |
| Folds | 5 |

![Fallback usage by fold](reports/figures/rung_usage_by_fold.png)

Flight-profile medians cover 59-82% of validation rows. Fewer than 1.5% fall through to the global median in any fold.

The second-level grouping was selected on the same folds used to report the score, which is mild selection bias. The held-out test set remains untouched.

## Usage

**1. Install.** Python 3.13.9. Versions are pinned in `requirements.txt`: pandas 2.3.3, NumPy 2.3.5, scikit-learn 1.7.2.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

`requirements-dev.txt` adds Jupyter. Use `requirements.txt` for runtime only.

**2. Download the data.** Select every field in the [BTS table](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) and download each month from 2024-01 to 2025-12. Roughly 677 MB compressed. Extract into:

```text
data/raw/data_for_2024/*.csv     12 files
data/raw/data_for_2025/*.csv     12 files
```

**3. Run the notebooks** from inside `notebooks/`, in this order.

| Notebook | Produces |
|---|---|
| `data_quality_audit.ipynb` | Counts, nulls, duplicates, month coverage, written up in `reports/data_quality_audit.md` |
| `data_cleaning.ipynb` | `data/interim/seattle_ontime_clean.csv` |
| `linear_baseline.ipynb` | Splits, folds, baseline score, encoding work |

The third needs the CSV from the second. `reports/figures/make_figures.py` regenerates the figures above.

**4. Check correctness.** There is no automated test suite yet. `linear_baseline.ipynb` asserts that fold train and validation dates never overlap, that the test period starts after the train pool ends, and that the two partitions sum to the full dataset. It also counts fallback usage per fold.

## Next Steps

- Train and score linear regression and random forest candidates
- Extract notebook logic into `src/` and add automated tests
- Build the two-flight comparison interface with unseen-carrier validation
- Evaluate the selected model once on the held-out test set

## Repository Layout

```text
notebooks/         Audit, cleaning, and baseline notebooks
reports/           Provenance, quality, and feature audits
reports/figures/   Figures above, regenerated by make_figures.py
data/raw/          Untracked BTS downloads
data/interim/      Cleaned SEA-origin dataset
```

## References

- [BTS Reporting Carrier On-Time Performance](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) — data source
- [scikit-learn `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — the "samples must be equally spaced" requirement behind the date-based split

## License

MIT, see [`LICENSE`](LICENSE). BTS data is not redistributed here; download it from TranStats and check the source's terms.
