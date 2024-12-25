charts_width: int = 800
table_height: int = 400
cols_perf: list[str] = ["date", "num_ads", "price", "type"]
cols_prices: list[str] = ["ticker"]
map_name_to_type: dict = {
    "Apartments for sale": "sales_flats",
    "Apartments for rent": "rentals_flats",
    "Houses for sale": "sales_houses",
}
time_strings = ["1W", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]
