# Autoregressive Trading Model Workflow

This repository includes a runnable Python workflow for building an autoregressive (AR) model for trading research. The implementation is in `src/trading_ar_model/ar_trading_model.py` and follows the requested eight-step process without requiring third-party packages.

## 1. Data collection

Prepare a CSV file containing historical, consistently sampled observations. At minimum, the file needs:

- a date/time column, named `Date` by default;
- a target variable, named `Close` by default.

Example command:

```bash
python -m src.trading_ar_model.ar_trading_model --csv data/sample_prices.csv --date-column Date --target-column Close
```

The repository includes `data/sample_prices.csv` so the command can be tested immediately. Replace it with your own sufficiently long historical dataset before using the model for research.

The target can be prices, returns, spreads, volatility, or another numeric trading variable. Modeling returns is often more appropriate than modeling raw prices because returns are more likely to be stationary.

## 2. Data exploration and visualisation

The pipeline saves SVG artifacts that can be opened in a browser:

- `01_time_series.svg` to inspect trend, regime changes, seasonality, missing periods, and outliers;
- `02_acf.svg` to inspect autocorrelation;
- `03_pacf.svg` to inspect partial autocorrelation.

The JSON report also records missing-value counts and simple z-score outlier diagnostics.

## 3. Data preprocessing

The script:

1. sorts observations by date;
2. parses numeric values while accepting common missing-value tokens;
3. fills missing values with linear interpolation and edge forward/back filling;
4. computes a lightweight stationarity score based on trend correlation;
5. applies first differencing when `--difference auto` is selected and the trend-correlation score is high.

The stationarity score is intentionally dependency-free. If you install statistical packages later, you can replace or supplement it with a formal Augmented Dickey-Fuller test.

## 4. Model specification

The script calculates ACF and PACF values up to `--max-lag`. It uses statistically significant ACF/PACF lags as diagnostics, then compares candidate AR lag orders with a validation tail from the training data. If multiple lag orders are close, it prefers the simpler model.

The current implementation models only the target series. Exogenous variables such as volume, market-index returns, macro indicators, or sentiment can improve the model but would require extending the model to ARX/ARIMAX/SARIMAX-style estimation.

## 5. Model estimation

The selected AR(p) model is estimated with ordinary least squares:

```text
y_t = intercept + phi_1*y_(t-1) + ... + phi_p*y_(t-p) + error_t
```

The coefficients, intercept, selected lag order, and differencing order are written to `metrics.json`.

## 6. Forecasting and validation

The pipeline performs a chronological train/test split. It fits the model on the training window, recursively forecasts the test window, and reports:

- Mean Absolute Error (MAE);
- Root Mean Squared Error (RMSE);
- selected lag order;
- ACF/PACF suggested lags;
- differencing order;
- stationarity scores before and after preprocessing.

Outputs are saved to:

- `metrics.json` for model specification, assumptions, diagnostics, validation results, and refinement ideas;
- `forecasts.csv` for actual values, forecasts, errors, and directional signals;
- `04_forecast_vs_actual.svg` for a visual comparison.

## 7. Model refinement

If validation performance is weak on new data streams, revisit preprocessing and specification:

- model returns instead of prices;
- remove or cap obvious bad ticks;
- change frequency and session filters before creating the CSV;
- add exogenous predictors;
- shorten or lengthen the training period;
- compare AR against ARMA/ARIMA/SARIMAX or machine-learning baselines;
- use walk-forward validation instead of a single holdout split.

## 8. Documentation and communication

Each run exports a JSON report documenting model configuration, stationarity assumptions, lag diagnostics, validation metrics, and artifact paths. Treat the generated `long`, `short`, and `flat` signals as research outputs only. Production trading systems also need transaction costs, slippage, risk limits, position sizing, execution constraints, and live monitoring.
