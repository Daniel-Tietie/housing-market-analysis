# Power BI Dashboard — Build Instructions

## Data source

Connect Power BI Desktop to the SQLite database at `data/housing.db`.
Use the ODBC connector or export `data/processed/ames_clean.csv` and load that directly (CSV is simpler and avoids driver setup).

**To load the CSV:**
1. Open Power BI Desktop
2. Get Data > Text/CSV > select `data/processed/ames_clean.csv`
3. Keep default column type detection

## Pages to build

### Page 1 — Overview
- Card visuals: total listings count, average sale price, median sale price
- Line chart: average sale price by `yr_sold`
- Bar chart: listing volume by `yr_sold`

### Page 2 — Geographic (Neighborhood)
- Horizontal bar chart: average sale price by `neighborhood`, sorted descending
- Add a slicer on `bldg_type` to filter by property type

### Page 3 — Trends
- Dual-axis chart: listing volume (bars) + average price (line) by `mo_sold`
- Label x-axis months as Jan–Dec
- Scatter: `gr_liv_area` vs `sale_price`, colored by `overall_qual`

### Page 4 — Risk / Volatility
- Table: neighborhood, listing count, avg price, calculated StdDev of price
- Coefficient of variation = StdDev / Avg (add as a DAX measure)
- Bar chart: neighborhoods ranked by CoV (most volatile at top)

## Design guidelines

- Color theme: two blues (primary `#2b6cb0`, accent `#90cdf4`) plus neutral gray `#718096`
- Remove default gridlines from all charts
- Font: Segoe UI, 11pt body, 13pt titles
- No 3D effects, no default Power BI color rainbow
- Export 3–4 page screenshots as PNG into `dashboard/screenshots/` and reference them in the README

## DAX measures to add

```dax
Avg Price = AVERAGE(ames_clean[sale_price])
Median Price = MEDIAN(ames_clean[sale_price])
Price StdDev = STDEV.P(ames_clean[sale_price])
Price CoV = DIVIDE([Price StdDev], [Avg Price])
Price Per SqFt = DIVIDE(AVERAGE(ames_clean[sale_price]), AVERAGE(ames_clean[gr_liv_area]))
```
