| Source | Dataset | Frequency | Coverage | Units | Access | Notes |
|---|---|---|---|---|---|---|
| FRED | PCOPPUSDM (Global Price of Copper) | Monthly (expected) | Historical (as provided by FRED CSV) | Retrieved from series page when possible; else unknown | Public CSV endpoint: https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOPPUSDM | Retrieved via public CSV endpoint; no API key required. |
| Yahoo | HG=F (COMEX copper futures) | Daily | Historical (as provided by Yahoo) | USD | Public CSV endpoint via Yahoo Finance download | Represents COMEX futures (HG=F), not LME cash price. Saved under `data/raw/yahoo/`. |
| USGS | Public NMIC pages (search) | Varies | Varies | Varies | Public, but CSV availability inconsistent | Script attempts to discover CSV links; may require manual selection. |
| LME | LME official data | Current-year (may be paywalled) | Limited public data | Varies | Likely paywalled for historical series | LME historical prices often require subscription; do not bypass paywalls. |

## Live ingestion run (2026-09-17)

- FRED `PCOPPUSDM`: Successfully retrieved via public CSV endpoint.
	- Raw file: `data/raw/fred/fred_PCOPPUSDM_1789654690.csv` (12076 bytes)
	- Manifest entry: `data/raw/fred_PCOPPUSDM_1789654690.csv.manifest.json` and global manifest `data/raw/manifest.json`
	- Frequency: inferred `MS` (month start). Units: not available from public CSV (best-effort lookup returned none).
	- Coverage: 1992-01-01 through 2026-07-01 (earliest/latest observed in file).


## USGS MCS 2026 investigation

- I checked the official USGS Mineral Commodity Summaries 2026 page (`https://pubs.usgs.gov/periodicals/mcs2026/`) for direct links to machine-readable files (`.csv`, `.xlsx`, `.xls`). None were found on the index page — the release is published primarily as PDFs. Because the canonical MCS release does not expose a structured CSV/XLSX for copper on the index, I did not implement a fragile PDF scraper. If you want structured MCS tables, we can either request official data from USGS or implement a robust PDF->table extraction workflow (separate task).

## Yahoo live retrieval note

- The repository includes a `YahooIngestor` adapter and offline tests that use a fixture CSV. In this environment I attempted to fetch `HG=F` from Yahoo Finance but received HTTP 429/401 responses (programmatic access appears blocked/restricted). Options:
  - Use the included adapter with a browser session or run with `yfinance` installed locally (more robust) on your machine.
  - Provide an alternate public data source or an API keyed service if programmatic access is required in CI.

- LME: public site check indicates no freely available CSV; LME pages are likely paywalled or require manual access.
	- Note: the run also hit an SSL cert verification error when contacting `lme.com` from this environment; even a simple discovery check failed due to certificate verification. Do not attempt to bypass paywalls.
