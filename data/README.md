# Data

Raw BTS downloads should be stored locally in `data/raw/` and kept out of Git.

Expected raw files are the monthly pre-zipped BTS downloads for:

- 2024-01 through 2024-12
- 2025-01 through 2025-12

After ingestion, the working dataset should be filtered to records where `Origin == "SEA"`.
