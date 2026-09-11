# Seattle Flight Reliability

Compare two Seattle-origin domestic flights using historical BTS on-time performance data.

## Project Status

Data provenance, quality audit, feature audit, and cleaning are complete. A clean interim dataset of 324,490 SEA-origin flights is in place, the temporal train/test split and cross-validation strategy are defined and verified, and the naive per-profile median baseline is built and scored. No trained model exists yet — the baseline is what any real model will need to beat.

## Problem

A traveler choosing between two SEA-departing domestic flights wants an honest comparison of historical arrival-delay reliability before booking.

The app will compare two flights using only information that is known before booking, such as carrier, destination, scheduled departure and arrival time, date features, and distance.

## Data Source

- Source: U.S. Department of Transportation, Bureau of Transportation Statistics, TranStats
- Table: Reporting Carrier On-Time Performance (1987-present)
- Period: full years 2024 and 2025, all 24 months verified present
- Scope: SEA-origin U.S. domestic flights
- Pull date: July 20, 2026

The raw pull covers all U.S. domestic flights: 14,080,680 rows across 110 columns. Filtering to SEA-origin leaves 328,559 rows. Dropping rows with a null `ArrDelay` leaves 324,490 rows in the clean interim dataset.

Raw BTS downloads are not tracked in Git because they are large. Place monthly zip files in `data/raw/`. The cleaned interim dataset is written to `data/interim/seattle_ontime_clean.csv`.

## Setup and Reproduction

**Environment.** Python 3.13.9, with the exact package versions the results were produced
under pinned in `requirements.txt` (`pandas` 2.3.3, `numpy` 2.3.5, `scikit-learn` 1.7.2).
The `scikit-learn` pin is deliberate: its model-persistence docs state there is no supported
way to load a model trained under a different version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs the runtime dependencies plus Jupyter. Use
`requirements.txt` alone if you only need the runtime set.

**Get the data.** The raw BTS files are not tracked in Git — 24 monthly CSVs total roughly
677 MB compressed. Download them yourself from the
[BTS TranStats Reporting Carrier On-Time Performance table](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr):
select every field, then download one month at a time for 2024-01 through 2025-12.

Unzip them into two directories, which is the layout the notebooks glob for:

```
data/raw/data_for_2024/*.csv    # 12 files, 2024-01 .. 2024-12
data/raw/data_for_2025/*.csv    # 12 files, 2025-01 .. 2025-12
```

**Run the notebooks in this order.** Each is run from inside `notebooks/`, since the paths
are relative to that directory.

| # | Notebook | Reads | Produces |
|---|---|---|---|
| 1 | `data_quality_audit.ipynb` | all 24 raw CSVs | row/column counts, null and duplicate checks, month-coverage check — written up in `reports/data_quality_audit.md` |
| 2 | `data_cleaning.ipynb` | all 24 raw CSVs | `data/interim/seattle_ontime_clean.csv` — SEA-origin rows with a non-null `ArrDelay` (324,490 rows) |
| 3 | `linear_baseline.ipynb` | `data/interim/seattle_ontime_clean.csv` | the temporal split, the 5 cross-validation folds, the 3-rung median baseline and its 19.95-minute MAE, and the feature-encoding work |

Notebook 3 depends on the interim CSV from notebook 2. Notebooks 1 and 2 both read the raw
files directly and can be run in either order.

**How correctness is checked.** There is **no standalone automated test suite** in this
repository — no `pytest`, no `tests/` directory. Correctness is currently enforced by
assertions inside `linear_baseline.ipynb`, which fail loudly if the split design breaks:

- `assert not set(train_dates) & set(val_dates)` — no date appears in both sides of a fold
- `assert train_pool['FlightDate'].max() < test['FlightDate'].min()` — the held-out test partition is strictly later than the train pool
- `assert len(train_pool) + len(test) == len(df)` — the split partitions the data with nothing lost or duplicated

Beyond those, the numeric claims in this README and in `reports/` were each computed against
the dataset rather than recalled, and the fold-level rung usage of the baseline is counted
per fold rather than assumed. Extracting the notebook logic into `src/` with a real test
suite is tracked in the backlog and has not been done yet.

## Modeling Guardrails

Allowed model features must be known before booking. Post-flight fields such as actual times, departure delay, taxi time, wheels-off/on time, air time, and delay-cause columns are excluded from model features.

Every one of the 110 raw columns was sorted into target, pre-booking, post-flight, or drop. The result is 13 pre-booking features plus the target. `Origin` was dropped because every row is SEA after filtering, so the column is constant. The full column-by-column reasoning is in `reports/feature_audit.md`.

`ArrDelay` is the regression target. Cancelled and diverted flights have a null `ArrDelay` because they never arrive, so those rows are dropped. This narrows the model's scope: it answers "given that this flight operates and arrives, what arrival delay should I expect?" It does not predict cancellation or diversion risk.

## Evaluation Strategy

The data is split by calendar date, not by row: a single cutoff of `2025-10-01` separates a **train pool** (2024-01-01 to 2025-09-30, 639 dates, 285,649 rows) from a held-out **test** partition (2025-10-01 to 2025-12-31, 92 dates, 38,841 rows, ~12%). Test is touched exactly once, at final evaluation — it is never used to compare or select models.

There is no separate fixed validation set. Every candidate model is scored on the same 5 `TimeSeriesSplit` folds over the train pool, so comparisons across models are apples-to-apples. The splitter is run over the array of unique dates in the train pool, not over rows, because flight rows are not evenly spaced (242–555 flights/day) while calendar dates are — splitting on rows would let a single date's flights land on both sides of a fold, which is exactly what this design avoids. `gap=0` is used deliberately: no feature here is lagged or rolling, so no constructed value could leak across a fold boundary.

One limitation worth stating plainly: because the split is temporal, the test partition falls entirely in Q4 (Oct–Dec). The final holdout number will describe winter operations, not a year-round average.

## Naive Baseline

The comparator every real model has to beat: a per-flight-profile historical median of `ArrDelay`, scored with the same 5 `TimeSeriesSplit` folds as every other candidate. Median, not mean, since 57.2% of SEA departures arrive early and a thin right tail (p99=154, max=3359) pulls the mean up — median is also the mathematically correct target when scoring with MAE.

Most flight profiles (carrier + flight number + destination) don't have enough training rows to trust a group median on their own, so predictions fall back through a 3-rung ladder: per-flight-profile median (needs ≥10 training rows), then carrier + scheduled-departure-hour median (also thresholded at ≥10), then the fold's flat global median. `n=10` was chosen by checking the thinnest fold: at that threshold only 5% of its training rows sit in profiles too thin to trust, versus 40% of individual profiles — most thin profiles carry little row weight. Rung usage is tracked and counted per fold rather than assumed, so the baseline can't quietly become "a global-median model wearing a per-profile label."

Rung 2's grouping was chosen by testing, not guessing: 11 candidate groupings were scored on the same folds, and carrier + departure-hour won clearly over the more obvious carrier + destination, since destination fragments groups without adding real signal while departure hour captures a genuine ~10-minute delay-propagation effect across the day. Mean cross-validated MAE is 19.95 minutes (std 1.21). A measured ceiling check — the best any median-based approach could theoretically do — shows only about 1.7 minutes of headroom above a flat global median, so a real model landing close to this number later is not necessarily a bug; `ArrDelay` itself is largely unpredictable from booking-time information alone.

## What Went Wrong

Three decisions that looked correct and were not. Each was caught by checking a number rather
than trusting the obvious reading.

**Cross-validation was splitting mid-day.** `TimeSeriesSplit` splits on array position, not on
time. Fed raw flight rows at 242-555 per day, all 5 fold boundaries landed inside a calendar
date, putting one day's flights on both sides of a train/validation split. The API reference
carries the qualifier that catches it: samples must be equally spaced. Flight rows are not,
calendar dates are. Running the splitter over the array of unique dates and mapping back with
`.isin()` fixed it. Before: validation folds spanned 108, 125, 136, 106 and 124 distinct days
at identical row counts. After: equal folds, zero date overlap.

**The baseline's middle rung was doing almost nothing.** Rung 2 originally grouped by carrier
and destination, and measured against the full ladder it looked fine — most rows never reach
rung 2 and contribute identical error to both sides, diluting the comparison. Scored on only
the rows that actually fall through, it was worth 0.149 minutes. Testing 11 candidate
groupings on the same folds put carrier and departure hour at 0.415, nearly triple.
Destination turned out to be actively harmful: carrier alone scored 0.210, and adding month
collapsed it to -0.001. Destination fragments groups without adding signal, while departure
hour carries a real 10-minute effect across the day. One caveat stated rather than hidden:
that grouping was selected on the same folds used for every other comparison, which is mild
selection bias. It is defensible only because a stronger baseline makes the later "does the
model beat it" test harder.

**One bug cross-validation is structurally unable to find.** With `drop="first"`, the dropped
reference category encodes as all zeros. With scikit-learn's default
`handle_unknown="ignore"`, an unseen category encodes as all zeros too. The vectors are
byte-identical, so an airline the model has never seen is silently predicted as whichever
carrier was dropped. The documented fix routes unknowns to a dedicated column, but only when
one exists, and `min_frequency` is a single threshold shared across every column. The rarest
carrier has 744 rows in the train pool, so no usable threshold ever pools one. Cross-validation
cannot surface this: 0 unseen carriers appear across all 5 validation folds. The bug never
fires on 2024-2025 data, but the app predicts 2026 onward, where a carrier starting SEA
service is ordinary — and because this tool compares two flights, an unrecognised carrier
returns a confident comparison in which one side is another airline's delay profile. No
encoder setting fixes it honestly, so it is closed as input validation at the app layer,
checked against the fitted encoder's `categories_` before predicting.

## Output

Done:

- Data provenance memo (`reports/data_provenance.md`)
- Data quality audit (`reports/data_quality_audit.md`)
- Feature availability audit (`reports/feature_audit.md`)
- Temporal train/test split and cross-validation strategy (see Evaluation Strategy above)
- Naive per-profile median baseline (see Naive Baseline above)

Planned:

- Linear regression candidate model
- Random forest candidate model
- Two-flight comparison interface
- Model documentation with limitations and honest project framing

## Repository Layout

- `notebooks/` — `data_quality_audit.ipynb`, `data_cleaning.ipynb`, `linear_baseline.ipynb`
- `reports/` — provenance, quality, and feature audit memos
- `data/raw/` — untracked BTS monthly downloads
- `data/interim/` — cleaned SEA-origin dataset

## Existing Consumer Tools

Consumer flight tools already provide some flight-reliability information. This project is being built for learning, deployment practice, and interview discussion, not because travelers lack any existing tools.

## License and Data Use

This project's code and documentation are released under the MIT License; see `LICENSE`.

The flight data is not redistributed here. It is published by the U.S. Department of
Transportation, Bureau of Transportation Statistics, and must be downloaded from TranStats
directly, as described in Setup and Reproduction above. Check the BTS site for the terms
that apply to its data before redistributing it.
