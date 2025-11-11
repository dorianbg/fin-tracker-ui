import duckdb

t = duckdb.sql("""
with st1 as (
    SELECT column1, column2, column3::double as pmt, column4 as pmt2
    FROM read_csv('research/foo/foo-page-*.csv', files_to_sniff = -1, union_by_name=true)
    WHERE column1 in ('YOURPARKINGSPACE L', 'RENTAL PAYMENT', 'JUSTPARK PARKING L', 'JUSTPARK WITHDRAWA', 'PARK LET LIMITED')
)
--select * from st1

select column1, sum(pmt) as payment, count(*) 
from st1
group by column1
""")
print(t)
#
# 9847.90 + 5930.29 = 15,778.19
# 857.97 + 10362.96 = 11,220.93
# 597.0
