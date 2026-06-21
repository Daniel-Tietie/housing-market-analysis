-- Key analytical queries for the housing market analysis.
-- Run from Python via: pd.read_sql(query, conn)

-- Average sale price by neighborhood, with listing count
SELECT
    neighborhood,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price,
    ROUND(MIN(sale_price), 0)       AS min_price,
    ROUND(MAX(sale_price), 0)       AS max_price,
    ROUND(AVG(gr_liv_area), 0)      AS avg_above_grade_sqft
FROM clean_housing
GROUP BY neighborhood
ORDER BY avg_price DESC;

-- Price trend by year sold
SELECT
    yr_sold,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price,
    ROUND(AVG(sale_price) / AVG(gr_liv_area), 2) AS avg_price_per_sqft
FROM clean_housing
GROUP BY yr_sold
ORDER BY yr_sold;

-- Price by overall quality rating
SELECT
    overall_qual,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price
FROM clean_housing
GROUP BY overall_qual
ORDER BY overall_qual;

-- Price by decade built
SELECT
    (year_built / 10) * 10          AS decade_built,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price,
    ROUND(AVG(age_at_sale), 1)      AS avg_age_at_sale
FROM clean_housing
GROUP BY decade_built
ORDER BY decade_built;

-- Impact of garage on price
SELECT
    has_garage,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price,
    ROUND(AVG(garage_cars), 1)      AS avg_garage_cars
FROM clean_housing
GROUP BY has_garage
ORDER BY has_garage DESC;

-- Distribution of sale prices in buckets
SELECT
    CASE
        WHEN sale_price <  100000 THEN 'Under 100k'
        WHEN sale_price <  150000 THEN '100-150k'
        WHEN sale_price <  200000 THEN '150-200k'
        WHEN sale_price <  250000 THEN '200-250k'
        WHEN sale_price <  300000 THEN '250-300k'
        WHEN sale_price <  400000 THEN '300-400k'
        ELSE '400k+'
    END                             AS price_range,
    COUNT(*)                        AS listings
FROM clean_housing
GROUP BY price_range
ORDER BY MIN(sale_price);

-- Seasonal patterns: average price and volume by month sold
SELECT
    mo_sold,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price
FROM clean_housing
GROUP BY mo_sold
ORDER BY mo_sold;

-- Top 10 most expensive neighborhoods by median price
SELECT
    neighborhood,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price
FROM clean_housing
GROUP BY neighborhood
HAVING COUNT(*) >= 10
ORDER BY avg_price DESC
LIMIT 10;

-- Price per square foot by building type
SELECT
    bldg_type,
    COUNT(*)                        AS listings,
    ROUND(AVG(sale_price), 0)       AS avg_price,
    ROUND(AVG(sale_price) / AVG(gr_liv_area), 2) AS avg_price_per_sqft
FROM clean_housing
GROUP BY bldg_type
ORDER BY avg_price DESC;

-- Correlation proxy: average price by living area quartile
SELECT
    NTILE(4) OVER (ORDER BY gr_liv_area)    AS area_quartile,
    COUNT(*)                                AS listings,
    ROUND(AVG(gr_liv_area), 0)              AS avg_sqft,
    ROUND(AVG(sale_price), 0)              AS avg_price
FROM clean_housing
GROUP BY area_quartile
ORDER BY area_quartile;
