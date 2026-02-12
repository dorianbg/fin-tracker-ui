import duckdb

t = duckdb.sql("""
with st1 as (
    SELECT *, case when "#ID" >= '444089838' then 'S' else 'D' end as t
    FROM read_csv('research/JustPark Transactions 3-Oct-25-11.08 copy.csv')
    where Description like 'Payment withdrawal' 
        --and "#ID" >= '444089838' 
)
select t, sum(replace("Money out", '-£','')::double) as m from st1
group by t
""")
print(t)
# justpark silvijo: 9847.90 + 5930.29 = 15,778.19
# justpark dorian: 856.92
# justpark total 16,635.11

# yourparkingspace silvijo: 857.97 + 10362.96 = 11,220.93
# yourparkingspace dorian: 500 at least
# 18229896 £105.85 Dorian Beganovic 67361068 309089
# 21823252 £136.00 Dorian Beganovic 67361068 309089
# 21823252 £136.00 Dorian Beganovic 67361068 309089
# 21823252 £136.00 Dorian Beganovic 67361068 309089

# parklet silvijo: 597.0
# parklet dorian: 98.40
# found both payments
# 30/01/2023 03/02/2023 05/02/2023 £60.00 £18.00 £0.00 £3.60 £38.40 Settled
# 26/07/2023 03/08/2023 05/08/2023 £85.71 £21.43 £0.00 £4.29 £60.00 Settled


t2 = duckdb.sql("""
WITH payments as (
    select "#ID" as ID, * from read_csv('research/JustPark Transactions 3-Oct-25-11.08 copy.csv')
),
parsed_payments AS (
    SELECT 
        ID,
        STRPTIME("Cleared Date", '%d/%b/%Y') AS cleared_date,
        -- Parse 'Money in' to numeric (remove '£' and ',', handle empty as 0)
        CASE 
            WHEN "Money in" = '' THEN 0 
            ELSE CAST(REPLACE(REPLACE("Money in", '£', ''), ',', '') AS DECIMAL(10, 2)) 
        END AS money_in_numeric,
        -- Parse 'Money out' to numeric (remove '£' and ',', handle empty as 0)
        CASE 
            WHEN "Money out" = '' THEN 0 
            ELSE CAST(REPLACE(REPLACE("Money out", '£', ''), ',', '') AS DECIMAL(10, 2)) 
        END AS money_out_numeric
    FROM payments
),
tax_year_assignments AS (
    SELECT 
        *,
        -- Calculate tax start year: if cleared_date >= April 6 of its year, use that year; else previous year
        CASE 
            WHEN cleared_date >= MAKE_DATE(YEAR(cleared_date), 4, 6) 
            THEN YEAR(cleared_date) 
            ELSE YEAR(cleared_date) - 1 
        END AS tax_start_year
    FROM parsed_payments
)
SELECT 
    -- Format tax year as 'YYYY/YY' (e.g., '2024/25')
    CAST(tax_start_year AS VARCHAR) || '/' || RIGHT(CAST(tax_start_year + 1 AS VARCHAR), 2) AS tax_year,
    COUNT(*) AS payment_count,
    SUM(money_in_numeric) AS total_money_in,
    SUM(money_out_numeric) AS total_money_out,
    SUM(money_in_numeric + money_out_numeric) AS net_total
FROM tax_year_assignments
GROUP BY tax_start_year
ORDER BY tax_start_year;
""")
print(t2)


t3 = duckdb.sql("""
WITH payments AS (
  SELECT "#ID" AS ID, * FROM read_csv('research/JustPark Transactions 3-Oct-25-11.08 copy.csv')
),
parsed_payments AS (
  SELECT 
    ID,
    STRPTIME("Cleared Date", '%d/%b/%Y') AS cleared_date,
    -- Parse 'Money in' to numeric (remove '£' and ',', handle empty as 0)
    CASE WHEN "Money in" = '' THEN 0 ELSE CAST(REPLACE(REPLACE("Money in", '£', ''), ',', '') AS DECIMAL(10, 2)) END AS money_in_numeric,
    -- Parse 'Money out' to numeric (remove '£' and ',', handle empty as 0)
    CASE WHEN "Money out" = '' THEN 0 ELSE CAST(REPLACE(REPLACE("Money out", '£', ''), ',', '') AS DECIMAL(10, 2)) END AS money_out_numeric,
    Description
  FROM payments
),
calendar_year_assignments AS (
  SELECT 
    *,
    YEAR(cleared_date) AS calendar_year
  FROM parsed_payments
)
SELECT 
  CAST(calendar_year AS VARCHAR) AS year,
  COUNT(distinct "ID") filter (where Description like 'Payment for booking%') AS payment_count,
  SUM(money_in_numeric) AS total_money_in,
  SUM(money_out_numeric) AS total_money_out,
  SUM(money_in_numeric + money_out_numeric) AS net_total
FROM calendar_year_assignments
GROUP BY calendar_year
ORDER BY calendar_year;
                """)
print(t3)
