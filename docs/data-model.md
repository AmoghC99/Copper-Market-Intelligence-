**Raw Layer**
- Location: `data/raw/`
- Contains original downloaded files (CSV/JSON) and per-file manifests (`.manifest.json`).

**Processed Layer**
- Location: `data/processed/`
- Contains canonical datasets used for downstream feature engineering and modeling.

**Canonical Schemas**

1) Market daily (daily financial time series)
- `timestamp` (datetime, UTC-aware recommended)
- `open` (float)
- `high` (float)
- `low` (float)
- `close` (float)
- `volume` (int or float)
- `source` (string) e.g. "Yahoo Finance"
- `instrument` (string) e.g. "COMEX copper futures (HG=F)"

2) FRED monthly benchmark
- `timestamp` (datetime at month granularity)
- `copper_benchmark_usd_t` (float) — price in USD per metric ton
- `source` (string) e.g. "FRED"
- `frequency` (string) e.g. "monthly"

**Data Types**
- Dates: ISO-8601 parseable strings, stored as datetime in processed datasets.
- Numeric prices: float; missing values preserved as NaN.
- Volume: integer where available; may be float if aggregated.

**Yahoo normalization rule (Stage 3)**
- During Yahoo `normalize_yahoo()` the ETL will drop rows that are fully blank/malformed where:
	- `timestamp` is null AND all market numeric fields (`open`, `high`, `low`, `close`, `volume`) are null.
- These rows are assumed to be completely malformed (e.g., stray blank/footer rows) and are removed before producing `market_daily.parquet`.
- Partially populated rows (e.g., a present timestamp but missing OHLC/volume, or numeric values with no timestamp) are NOT silently dropped; they are retained and reported in validation outputs for further inspection.
- The ETL report includes `rows_in`, `rows_removed`, and `rows_out` for each dataset to record removals.

**Sources & Notes**
- `FRED PCOPPUSDM`: monthly global price of copper (USD per metric ton). Retrieved from FRED public CSV.
- `Yahoo HG=F`: COMEX copper futures daily OHLCV. Continuous futures may show contract-roll effects; treat as proxy for short-term market prices.

**Known Limitations**
- Yahoo programmatic CSV endpoints may be rate-limited; prefer `yfinance` for reproducible downloads.
- USGS MCS 2026 official index does not expose structured CSV/XLSX for copper in the index; MCS is primarily PDFs. Consider requesting official data or applying a validated PDF extraction workflow if needed.
