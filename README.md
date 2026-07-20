# Seattle Flight Reliability

Compare two Seattle-origin domestic flights using historical BTS on-time performance data.

## Project Status

This project is in the data provenance and quality audit stage. No model has been trained yet.

## Problem

A traveler choosing between two SEA-departing domestic flights wants an honest comparison of historical arrival-delay reliability before booking.

The app will compare two flights using only information that is known before booking, such as carrier, destination, scheduled departure and arrival time, date features, and distance.

## Data Source

- Source: U.S. Department of Transportation, Bureau of Transportation Statistics, TranStats
- Table: Reporting Carrier On-Time Performance (1987-present)
- Planned period: full years 2024 and 2025
- Planned scope: SEA-origin U.S. domestic flights
- Pull date: July 20, 2026

Raw BTS downloads are not tracked in Git because they are large. Place monthly zip files in `data/raw/`.

## Modeling Guardrails

Allowed model features must be known before booking. Post-flight fields such as actual times, departure delay, taxi time, wheels-off/on time, air time, and delay-cause columns are excluded from model features.

`ArrDelay` is the regression target. Cancelled and diverted flights are excluded from the arrival-delay estimate but reported separately as historical context.

## Planned Output

- Data provenance and quality memo
- Baseline model
- Random forest candidate model
- Two-flight comparison interface
- README/model documentation with limitations and honest project framing

## Existing Consumer Tools

Consumer flight tools already provide some flight-reliability information. This project is being built for learning, deployment practice, and interview discussion, not because travelers lack any existing tools.
