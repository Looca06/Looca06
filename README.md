# Looca06

- 👋 Hi, I’m @Looca06
- 👀 I’m interested in AI, Trading
- 🌱 I’m currently learning Math, Science
- 💞️ I’m looking to collaborate on AI and trading research projects
- 📫 How to reach me: add your preferred contact method here
- 😄 Pronouns: add your pronouns here
- ⚡ Fun fact: I enjoy exploring quantitative finance ideas

## Autoregressive model for trading research

This repository now includes a Python workflow for building and validating an autoregressive (AR) model on historical trading time-series data.

### What the workflow does

1. Loads historical CSV data with a date column and numeric target column.
2. Visualizes the raw time series plus ACF/PACF diagnostics as SVG files.
3. Handles missing values and checks stationarity with a dependency-free trend diagnostic.
4. Applies first differencing automatically when needed.
5. Selects the AR lag order using ACF/PACF diagnostics and validation RMSE.
6. Splits data chronologically into train and test sets.
7. Forecasts the test window and reports MAE/RMSE.
8. Saves documentation-ready metrics, plots, forecasts, and simple long/short/flat research signals.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
# No third-party dependencies are required.
# requirements.txt documents that the model uses only the Python standard library.
```

### Run

```bash
python -m src.trading_ar_model.ar_trading_model \
  --csv data/sample_prices.csv \
  --date-column Date \
  --target-column Close \
  --output-dir artifacts/sample_ar_model
```

Replace `data/sample_prices.csv` with your own historical trading CSV when ready.

For detailed methodology, assumptions, outputs, and refinement ideas, see [`docs/autoregressive_trading_model.md`](docs/autoregressive_trading_model.md).

> This project is for education and research only. It is not financial advice.

<!---
Looca06/Looca06 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
