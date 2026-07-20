# Data Provenance And Quality Memo

## Source

- Agency: U.S. Department of Transportation, Bureau of Transportation Statistics
- Platform: TranStats
- Database: On-Time
- Table: Reporting Carrier On-Time Performance (1987-present)
- Latest available data shown by TranStats: May 2026
- Pull date: July 20, 2026

## Planned Extract

- Years: 2024 and 2025
- Periods: January through December for each year
- Geography filter: All
- Later project filter: `Origin == "SEA"`
- Raw file type: BTS monthly pre-zipped CSV downloads

## Field Notes

The raw table contains both pre-booking fields and post-flight fields. Post-flight fields may be useful for audit and target construction, but they must not be used as model features.

Target field:

- `ArrDelay`

Context fields:

- `ArrDel15`
- `Cancelled`
- `Diverted`

Candidate pre-booking fields:

- `Reporting_Airline`
- `Flight_Number_Reporting_Airline`
- `Origin`
- `Dest`
- `CRSDepTime`
- `CRSArrTime`
- `DayOfWeek`
- `Month`
- `Distance`

Excluded from model features:

- actual departure or arrival times
- `DepDelay` and related departure-delay fields
- taxi, wheels-off/on, and air-time fields
- delay-cause fields
- diverted-airport outcome fields

## Quality Checks To Complete

- Confirm all 24 monthly files are present.
- Confirm row counts for each month.
- Confirm SEA-origin row counts for each month.
- Check missingness for target and context fields.
- Count cancelled and diverted flights separately.
- Confirm the date range is complete and chronological.
- Decide whether flight number adds useful signal or creates instability.
