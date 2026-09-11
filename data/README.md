# Data

Raw BTS downloads should be stored locally in `data/raw/` and kept out of Git.

Expected raw files are the monthly pre-zipped BTS downloads for 2024-01 through 2025-12,
unzipped into the two directories the notebooks glob for:

- `data/raw/data_for_2024/` — 12 CSVs, 2024-01 through 2024-12
- `data/raw/data_for_2025/` — 12 CSVs, 2025-01 through 2025-12

After ingestion, the working dataset should be filtered to records where `Origin == "SEA"`.
