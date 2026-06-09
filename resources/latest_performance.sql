create or replace view total_return as
with prices as (
    select
        h.ticker,
        h.ticker_full,
        h."date" at time zone 'Europe/Paris' as date,
        h.open,
        h.high,
        h.low,
        h.close,
        h.volume,
        h.dividends,
        t.currency,
        t.description,
        t.fund_type,
        t.sector
    from historical_prices h
    join ticker_ref t on h.ticker_full = t.ticker_full
    where h.ticker<>'EURGBP=X'
      and h.ticker<>'GBP=X'
      and h.volume > 0
      and h.close is not null
      and h.close > 0
), exchange_rates as (
    select date at time zone 'Europe/Paris' as date,
           case when ticker = 'EURGBP=X' then 'EUR'
                when ticker = 'GBP=X' then 'USD'
                else 'GBP'
               end as ccy,
            "close" as close
    from historical_prices
    where ticker='EURGBP=X' or ticker='GBP=X'
), prices_w_exchange_rate as (
        select prices.ticker,
                prices.ticker_full,
                prices."date",
                prices."open",
                prices.high,
                prices.low,
                prices."close",
                prices.volume,
                prices.dividends,
                prices.currency,
                prices.description,
                prices.fund_type,
                prices.sector,
                coalesce(exchange_rates.ccy, 'GBP') as ccy2,coalesce(exchange_rates.close, 1) as ccy_conversion
        from prices
        ASOF LEFT JOIN exchange_rates
            on prices.currency = exchange_rates.ccy and prices.date >= exchange_rates.date
)
select ticker,
       ticker_full,
       "date",
       "open" as open_orig,
       high as high_orig,
       low as low_orig,
       "close" as price_orig,
       "open" * ccy_conversion as price_open,
       high * ccy_conversion as price_high,
       low * ccy_conversion as price_low,
       "close" * ccy_conversion as price_close,
       "close" * ccy_conversion as price,
       volume,
--        dividends * ccy_conversion as dividend_conv,
       currency,
       'GBP' as currency_converted,
       description,
       fund_type,
       sector
from prices_w_exchange_rate;


create or replace view instrument_annualised_volatility as
WITH log_returns AS (
    SELECT
        ticker_full,
        "date",
        "price",
        LN(h.price / LAG(h.price) OVER (PARTITION BY h.ticker ORDER BY date)) AS log_return
    FROM total_return h
),
-- Step 2: Calculate Daily Volatility
daily_volatility AS (
    SELECT
        ticker_full,
        STDDEV_POP(log_return) over (
            partition by ticker_full
            ORDER BY "date"
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) AS daily_volatility,
        "date"
    FROM log_returns
)
-- Step 3: Annualize Volatility
SELECT
    ticker_full,
    "date"::date as date,
    round(daily_volatility * 100, 2) as vol_1d,
    round(daily_volatility * SQRT(252) * 100, 2) as vol_1y
FROM
    daily_volatility;


create or replace view instrument_monthly_volatility as
WITH log_returns AS (
    SELECT
        ticker_full,
        "date",
        "price",
        LN(h.price / LAG(h.price) OVER (PARTITION BY h.ticker ORDER BY date)) AS log_return
    FROM total_return h
),
-- Step 2: Calculate Daily Volatility
daily_volatility AS (
    SELECT
        ticker_full,
        STDDEV_POP(log_return) over (
            partition by ticker_full
            ORDER BY "date"
            ROWS BETWEEN 21 PRECEDING AND CURRENT ROW
        ) AS daily_volatility,
        "date"
    FROM log_returns
)
-- Step 3: Annualize Volatility
SELECT
    ticker_full,
    "date"::date as date,
    round(daily_volatility * 100, 2) as vol_1d,
    round(daily_volatility * SQRT(21) * 100, 2) as vol_1mo
FROM
    daily_volatility;


create or replace view latest_performance as
with stage1 as (
    select
        ticker,
        h.ticker_full,
        "date",
        "price",
        description,
        fund_type,
        sector,
        "price" - (lag("price", 1, 0) over one_year) as day_price_diff,
        "price" - (lag("price", 5, 0) over one_year) as week_price_diff,
        "price" - (lag("price", 10, 0) over one_year) as two_week_price_diff,
        "price" - (lag("price", 15, 0) over one_year) as three_week_price_diff,
        "price" - (lag("price", 21, 0) over one_year) as month_price_diff,
        "price" - (lag("price", 63, 0) over one_year) as quarter_price_diff,
        "price" - (lag("price", 126, 0) over one_year) as half_year_price_diff,
        "price" - (lag("price", 252, 0) over one_year) as year_price_diff,
        "price" - (lag("price", 2*252, 0) over one_year) as two_year_price_diff,
        "price" - (lag("price", 3*252, 0) over one_year) as three_year_price_diff,
        "price" - (lag("price", 5*252, 0) over one_year) as five_year_price_diff
    from total_return h
    WINDOW
        one_year AS (
            PARTITION BY h.ticker_full
            ORDER BY "date" ASC
            RANGE BETWEEN INTERVAL 1890 DAYS PRECEDING AND current row
        )
), stage2 as (
SELECT
    ticker,
    ticker_full,
    "date"::date as date,
    "date" as dt,
    row_number() over (partition by ticker order by date desc) as rown,
    round("price", 2) as price,
    replace(replace(replace(description, 'iShares ', ''), 'MSCI ', ''), 'SPDR® ', '')  as description,
    fund_type,
    sector,
-- returns
    round(("price" / ("price" - day_price_diff) - 1) * 100, 2) as r_1d,
    round(("price" / ("price" - week_price_diff) - 1) * 100, 2) as r_1w,
    round(("price" / ("price" - two_week_price_diff) - 1) * 100, 2) as r_2w,
    round(("price" / ("price" - month_price_diff) - 1) * 100, 2) as r_1mo,
    round(("price" / ("price" - quarter_price_diff) - 1) * 100, 2) as r_3mo,
    round(("price" / ("price" - half_year_price_diff) - 1) * 100, 2) as r_6mo,
    round(("price" / ("price" - year_price_diff) - 1) * 100, 2) as r_1y,
    round(("price" / ("price" - two_year_price_diff) - 1) * 100, 2) as r_2y,
    round(("price" / ("price" - three_year_price_diff) - 1) * 100, 2) as r_3y,
    round(("price" / ("price" - five_year_price_diff) - 1) * 100, 2) as r_5y,
-- z-scores
    round((abs(day_price_diff) / (stddev_pop(day_price_diff) over one_month)), 2) as z_1d,
    round((abs(week_price_diff) / (stddev_pop(week_price_diff) over one_month)), 2) as z_1w,
    round(abs(two_week_price_diff) / (stddev_pop(two_week_price_diff) over one_month), 2) as z_2w,
    round(abs(month_price_diff) / (stddev_pop(month_price_diff) over one_month), 2) as z_1mo,
-- moving averages
    round(("price"/avg("price") over one_month - 1) * 100, 2) as ma_21,
    round(("price"/avg("price") over three_month - 1) * 100, 2) as ma_63,
    round(("price"/avg("price") over six_month - 1) * 100, 2) as ma_126,
    round(("price"/avg("price") over one_year - 1) * 100, 2) as ma_252,
-- drawdown from 52-week high
    round(("price" / max("price") over one_year - 1) * 100, 2) as drawdown_52w,
-- drawdown from 3-year high
    round(("price" / max("price") over three_year - 1) * 100, 2) as drawdown_3y,
-- range position: 0 = at period low, 100 = at period high
    round(
        ("price" - min("price") over one_year)
        / nullif(max("price") over one_year - min("price") over one_year, 0)
        * 100, 1) as range_pos_52w,
    round(
        ("price" - min("price") over two_year)
        / nullif(max("price") over two_year - min("price") over two_year, 0)
        * 100, 1) as range_pos_104w,
    round(
        ("price" - min("price") over three_year)
        / nullif(max("price") over three_year - min("price") over three_year, 0)
        * 100, 1) as range_pos_156w
FROM stage1
WINDOW
    one_year AS (
        PARTITION BY ticker_full
        ORDER BY "date" ASC
        RANGE BETWEEN INTERVAL 252 DAYS PRECEDING AND current row
    ),
    one_month AS (
        PARTITION BY ticker_full
        ORDER BY "date" ASC
        RANGE BETWEEN INTERVAL 21 DAYS PRECEDING AND current row
    ),
    three_month AS (
            PARTITION BY ticker_full
            ORDER BY "date" ASC
            RANGE BETWEEN INTERVAL 63 DAYS PRECEDING AND current row
    ),
    six_month AS (
            PARTITION BY ticker_full
            ORDER BY "date" ASC
            RANGE BETWEEN INTERVAL 126 DAYS PRECEDING AND current row
    ),
    two_year AS (
            PARTITION BY ticker_full
            ORDER BY "date" ASC
            RANGE BETWEEN INTERVAL 504 DAYS PRECEDING AND current row
    ),
    three_year AS (
            PARTITION BY ticker_full
            ORDER BY "date" ASC
            RANGE BETWEEN INTERVAL 756 DAYS PRECEDING AND current row
    )
)
select
    s.*,
    vol.vol_1d as vol_1d,
    vol.vol_1y,
    vol2.vol_1d as vol_1d_m,
    vol2.vol_1mo
from stage2 s
left join instrument_annualised_volatility vol
    on s.ticker_full = vol.ticker_full and s.date = vol.date
left join instrument_monthly_volatility vol2
    on s.ticker_full = vol2.ticker_full and s.date = vol2.date
;

create or replace view latest_performance_sharpe as
select
    s3.*,
    round((r_1d - r_1d_rf)/vol_1y, 2) as r_1d_s,
    round((r_1w - r_1w_rf)/vol_1y, 2) as r_1w_s,
    round((r_2w - r_2w_rf)/vol_1y, 2) as r_2w_s,
    round((r_1mo - r_1mo_rf)/vol_1y, 2) as r_1mo_s,
    round((r_3mo - r_3mo_rf)/vol_1y, 2) as r_3mo_s,
    round((r_6mo - r_6mo_rf)/vol_1y, 2) as r_6mo_s,
    round((r_1y - r_1y_rf)/vol_1y, 2) as r_1y_s,
    round((r_2y - r_2y_rf)/vol_1y, 2) as r_2y_s,
    round((r_3y - r_3y_rf)/vol_1y, 2) as r_3y_s,
    round((r_5y - r_5y_rf)/vol_1y, 2) as r_5y_s
from latest_performance as s3
ASOF LEFT JOIN (
--     risk free rate
    select
        date,
        r_1d as r_1d_rf,
        r_1w as r_1w_rf,
        r_2w as r_2w_rf,
        r_1mo as r_1mo_rf,
        r_3mo as r_3mo_rf,
        r_6mo as r_6mo_rf,
        r_1y as r_1y_rf,
        r_2y as r_2y_rf,
        r_3y as r_3y_rf,
        r_5y as r_5y_rf
    from latest_performance
    where ticker = 'CSH2'
) s4
on s3.date >= s4.date
order by s3.dt desc;
