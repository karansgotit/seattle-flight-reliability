# Seattle Flight Reliability

Compare two Seattle-origin domestic flights using historical BTS on-time performance data.

## Project Status

Data provenance, quality audit, feature audit, and cleaning are complete. A clean interim dataset of 324,490 SEA-origin flights is in place and the baseline modelling notebook is set up with the feature set selected. No model has been trained yet.

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

## Modeling Guardrails

Allowed model features must be known before booking. Post-flight fields such as actual times, departure delay, taxi time, wheels-off/on time, air time, and delay-cause columns are excluded from model features.

Every one of the 110 raw columns was sorted into target, pre-booking, post-flight, or drop. The result is 13 pre-booking features plus the target. `Origin` was dropped because every row is SEA after filtering, so the column is constant. The full column-by-column reasoning is in `reports/feature_audit.md`.

`ArrDelay` is the regression target. Cancelled and diverted flights have a null `ArrDelay` because they never arrive, so those rows are dropped. This narrows the model's scope: it answers "given that this flight operates and arrives, what arrival delay should I expect?" It does not predict cancellation or diversion risk.

## Output

Done:

- Data provenance memo (`reports/data_provenance.md`)
- Data quality audit (`reports/data_quality_audit.md`)
- Feature availability audit (`reports/feature_audit.md`)

Planned:

- Baseline model
- Random forest candidate model
- Two-flight comparison interface
- Model documentation with limitations and honest project framing

## Repository Layout

- `notebooks/` — `data_quality_audit.ipynb`, `data_cleaning.ipynb`, `linear_baseline.ipynb`
- `reports/` — provenance, quality, and feature audit memos
- `data/raw/` — untracked BTS monthly downloads
- `data/interim/` — cleaned SEA-origin dataset
- `PROJECT_ISSUES.md` — remaining work, tracked as issues
- `LEARNINGS.md` — notes on what was learned and where the reasoning went wrong

## Existing Consumer Tools

Consumer flight tools already provide some flight-reliability information. This project is being built for learning, deployment practice, and interview discussion, not because travelers lack any existing tools.
