import camelot

tables = camelot.read_pdf(
    "01_09_2024_31_08_2025 all copy.pdf", pages="all", flavor="network"
)
print(tables)
tables
tables.export("foo.csv", f="csv", compress=True)  # json, excel, html, markdown, sqlite
# tables[0]
# tables[0].parsing_report
# tables[0].to_csv("foo.csv")  # to_json, to_excel, to_html, to_markdown, to_sqlite
# tables[0].df  # get a pandas DataFrame!
