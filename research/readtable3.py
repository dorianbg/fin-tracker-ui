import duckdb

test_justpark_matches = duckdb.sql("""
SELECT *
FROM read_xlsx('/Users/dbg/Downloads/All parking receipts.xlsx') as t1
left join (
    select
        CASE 
            WHEN "Money out" = '' THEN 0 
            ELSE CAST(REPLACE(REPLACE("Money out", '£', ''), ',', '') AS DECIMAL(10, 2)) 
        END AS money_out_numeric,
        *
    from read_csv('research/JustPark Transactions 3-Oct-25-11.08 copy.csv')
    where description like 'Payment withdrawal'
) as t2 
on t1."Amount" = -1 * t2.money_out_numeric
where t1.Platform = 'JustPark'
    and money_out_numeric is not null
""")
print(test_justpark_matches)


distinct_bookings = duckdb.sql("""
SELECT 
    --*,
 count(distinct "Booking ID") as id
from read_csv('research/JustPark Earnings 3-Oct-25-11.08.csv')
""")
print(distinct_bookings)
