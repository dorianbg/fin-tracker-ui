# Fin Tracker UI

Streamlit application utilising DuckDB for querying in-memory data (load
ed from encrypted parquet files) alongside heavy caching due to the reactive architecture of Streamlit.

### Development

For local setup you should just `poetry install` the project.

Consider setting up `pyenv` and `poetry` as per this [article](https://dorianbg.github.io/posts/python-project-setup-best-practice/).

After installing dependencies above, you could run locally with:  
```python -m streamlit run app.py``` .

With PyCharm (or IntelliJ Idea) you can also utilise the debugger.
Just make sure that inside your `Run configuration` module is set to `streamlit` and script parameters are set to `run app.py`.

### Data

Dataset is easily generated from [this repo](https://github.com/dorianbg/fin-tracker/).

The committed parquet files are encrypted, so you won't be able to re-use the full dataset locally.
Instead us the `*_unencrypted.parquet` files to get a high level sense of the data.   

Or consider simply running the script from above mentioned repo.

### Screenshots

General performance tab:
![Screenshot 2024-02-29 at 23.08.08.jpg](img%2FScreenshot%202024-02-29%20at%2023.08.08.jpg)

Performance plotting tab:
![Screenshot 2024-02-29 at 23.07.50.jpg](img%2FScreenshot%202024-02-29%20at%2023.07.50.jpg)