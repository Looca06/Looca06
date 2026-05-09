"""Build and validate an autoregressive trading model with only the Python standard library.

The module implements the requested research workflow end to end:

1. Collect historical time-series data from a CSV file.
2. Explore the data with summary diagnostics and SVG plots.
3. Preprocess missing values and optionally difference non-stationary prices.
4. Specify an AR(p) model using ACF/PACF diagnostics and validation error.
5. Estimate the model with ordinary least squares.
6. Forecast a chronological test set and report MAE/RMSE.
7. Export artifacts that make refinement easier.
8. Document assumptions, limitations, validation results, and signals.

The implementation deliberately avoids third-party packages so it can run in restricted
environments. It is intended for education and research, not financial advice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Observation:
    """A single timestamped market observation."""

    date: str
    value: float | None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the autoregressive trading model."""

    csv_path: Path
    date_column: str = "Date"
    target_column: str = "Close"
    output_dir: Path = Path("artifacts")
    max_lag: int = 20
    test_size: float = 0.2
    difference: str = "auto"
    signal_threshold: float = 0.0
    validation_lag_tolerance: float = 0.05


@dataclass(frozen=True)
class ARModel:
    """Estimated AR(p) model coefficients."""

    intercept: float
    coefficients: list[float]

    @property
    def lag_order(self) -> int:
        """Return the AR lag order."""

        return len(self.coefficients)


@dataclass(frozen=True)
class ModelResults:
    """Validation and specification results for a trained AR model."""

    lag_order: int
    differencing_order: int
    train_observations: int
    test_observations: int
    mae: float
    rmse: float
    acf_suggested_lag: int
    pacf_suggested_lag: int
    stationarity_score_before: float
    stationarity_score_after: float
    forecast_path: str
    metrics_path: str


def parse_number(value: str) -> float | None:
    """Parse a numeric CSV field, accepting common missing-value tokens."""

    cleaned = value.strip().replace(",", "")
    if cleaned.lower() in {"", "na", "n/a", "null", "none", "nan"}:
        return None
    return float(cleaned)


def parse_sort_key(date_text: str) -> tuple[int, str]:
    """Return a sortable key for common trading date formats."""

    formats = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S")
    for date_format in formats:
        try:
            return (0, datetime.strptime(date_text, date_format).isoformat())
        except ValueError:
            continue
    return (1, date_text)


def load_observations(config: ModelConfig) -> list[Observation]:
    """Load sorted observations from a CSV file."""

    with config.csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
        missing = {config.date_column, config.target_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        observations = [
            Observation(row[config.date_column], parse_number(row.get(config.target_column, "")))
            for row in reader
        ]

    if not observations:
        raise ValueError("CSV file contains no observations.")
    return sorted(observations, key=lambda item: parse_sort_key(item.date))


def interpolate_missing(values: Sequence[float | None]) -> tuple[list[float], int]:
    """Fill missing values with linear interpolation plus edge forward/back filling."""

    filled: list[float | None] = list(values)
    missing_count = sum(value is None for value in filled)
    known_indices = [index for index, value in enumerate(filled) if value is not None]
    if not known_indices:
        raise ValueError("Target column has no numeric observations.")

    first_known = known_indices[0]
    for index in range(first_known):
        filled[index] = filled[first_known]

    last_known = known_indices[-1]
    for index in range(last_known + 1, len(filled)):
        filled[index] = filled[last_known]

    for left, right in zip(known_indices, known_indices[1:]):
        left_value = filled[left]
        right_value = filled[right]
        if left_value is None or right_value is None:
            raise ValueError("Unexpected missing interpolation endpoint.")
        gap = right - left
        for offset in range(1, gap):
            weight = offset / gap
            filled[left + offset] = left_value + (right_value - left_value) * weight

    return [float(value) for value in filled if value is not None], missing_count


def difference(values: Sequence[float]) -> list[float]:
    """Return first differences of a series."""

    return [current - previous for previous, current in zip(values, values[1:])]


def stationarity_score(values: Sequence[float]) -> float:
    """Estimate non-stationarity with trend and mean-shift diagnostics.

    This is a lightweight diagnostic rather than a formal ADF test. Scores near zero
    suggest weaker trend behavior; high scores suggest trend/non-stationarity.
    The score combines time correlation with a first-half/second-half mean shift so
    slow random-walk drift is more likely to trigger differencing.
    """

    if len(values) < 3 or len(set(round(value, 12) for value in values)) <= 1:
        return 0.0
    trend_score = abs(correlation(list(range(len(values))), list(values)))
    midpoint = len(values) // 2
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    mean_shift_score = abs(mean(values[:midpoint]) - mean(values[midpoint:])) / std_dev if std_dev else 0.0
    return min(1.0, max(trend_score, mean_shift_score))


def prepare_series(values: Sequence[float], mode: str) -> tuple[list[float], int, float, float]:
    """Return a stationary modeling series and before/after stationarity diagnostics."""

    if mode not in {"auto", "none", "first"}:
        raise ValueError("difference must be one of: auto, none, first")
    before = stationarity_score(values)
    if mode == "first" or (mode == "auto" and before > 0.30):
        prepared = difference(values)
        diff_order = 1
    else:
        prepared = list(values)
        diff_order = 0
    after = stationarity_score(prepared)
    if len(prepared) < 12:
        raise ValueError("At least 12 prepared observations are required to fit and test an AR model.")
    return prepared, diff_order, before, after


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean."""

    return sum(values) / len(values)


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Pearson correlation for two equal-length vectors."""

    if len(left) != len(right):
        raise ValueError("Correlation vectors must have the same length.")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_denominator * right_denominator
    return numerator / denominator if denominator else 0.0


def autocorrelation(values: Sequence[float], lag: int) -> float:
    """Calculate autocorrelation at a given lag."""

    if lag <= 0:
        return 1.0
    if lag >= len(values):
        raise ValueError("Lag must be smaller than the number of observations.")
    return correlation(values[:-lag], values[lag:])


def acf_values(values: Sequence[float], max_lag: int) -> list[float]:
    """Calculate ACF values from lag 1 through max_lag."""

    capped = min(max_lag, len(values) - 2)
    return [autocorrelation(values, lag) for lag in range(1, capped + 1)]


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a linear system with Gaussian elimination and ridge stabilization."""

    size = len(vector)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        if abs(pivot) < 1e-12:
            augmented[pivot_index][pivot_index] += 1e-8
            pivot = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]
    return [augmented[row][size] for row in range(size)]


def partial_autocorrelation(values: Sequence[float], lag: int) -> float:
    """Estimate PACF at a lag by fitting an AR(lag) Yule-Walker system."""

    if lag <= 0:
        return 1.0
    acf = [autocorrelation(values, current_lag) for current_lag in range(lag + 1)]
    matrix = [[acf[abs(row - column)] for column in range(lag)] for row in range(lag)]
    vector = [acf[row] for row in range(1, lag + 1)]
    return solve_linear_system(matrix, vector)[-1]


def pacf_values(values: Sequence[float], max_lag: int) -> list[float]:
    """Calculate PACF values from lag 1 through max_lag."""

    capped = min(max_lag, len(values) - 2)
    return [partial_autocorrelation(values, lag) for lag in range(1, capped + 1)]


def suggested_lag_from_correlation(correlations: Sequence[float], sample_size: int) -> int:
    """Choose the last significant correlation lag using a 95% white-noise band."""

    if not correlations:
        return 1
    significance_band = 1.96 / math.sqrt(sample_size)
    significant_lags = [index + 1 for index, value in enumerate(correlations) if abs(value) > significance_band]
    return max(significant_lags) if significant_lags else 1


def split_train_test(values: Sequence[float], test_size: float) -> tuple[list[float], list[float]]:
    """Split observations chronologically into training and test windows."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split_at = int(len(values) * (1 - test_size))
    train = list(values[:split_at])
    test = list(values[split_at:])
    if len(train) < 10 or len(test) < 1:
        raise ValueError("Not enough observations for the requested train/test split.")
    return train, test


def fit_ar_model(values: Sequence[float], lag_order: int) -> ARModel:
    """Estimate an AR(p) model with ordinary least squares."""

    if lag_order < 1:
        raise ValueError("lag_order must be at least 1")
    if len(values) <= lag_order:
        raise ValueError("lag_order must be smaller than the number of training observations")

    rows = []
    targets = []
    for index in range(lag_order, len(values)):
        rows.append([1.0] + [values[index - lag] for lag in range(1, lag_order + 1)])
        targets.append(values[index])

    column_count = lag_order + 1
    xtx = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    xty = [0.0 for _ in range(column_count)]
    for row, target in zip(rows, targets):
        for i in range(column_count):
            xty[i] += row[i] * target
            for j in range(column_count):
                xtx[i][j] += row[i] * row[j]
    for i in range(column_count):
        xtx[i][i] += 1e-8

    solution = solve_linear_system(xtx, xty)
    return ARModel(intercept=solution[0], coefficients=solution[1:])


def one_step_predictions(values: Sequence[float], model: ARModel) -> list[float]:
    """Generate in-sample one-step predictions after the warm-up lags."""

    predictions = []
    for index in range(model.lag_order, len(values)):
        prediction = model.intercept + sum(
            coefficient * values[index - lag]
            for lag, coefficient in enumerate(model.coefficients, start=1)
        )
        predictions.append(prediction)
    return predictions


def forecast_ar(history: Sequence[float], model: ARModel, steps: int) -> list[float]:
    """Forecast future values recursively from the latest known history."""

    history_values = list(history)
    forecasts = []
    for _ in range(steps):
        prediction = model.intercept + sum(
            coefficient * history_values[-lag]
            for lag, coefficient in enumerate(model.coefficients, start=1)
        )
        forecasts.append(prediction)
        history_values.append(prediction)
    return forecasts


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean Absolute Error."""

    return sum(abs(observed - forecast) for observed, forecast in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root Mean Squared Error."""

    return math.sqrt(sum((observed - forecast) ** 2 for observed, forecast in zip(actual, predicted)) / len(actual))


def select_lag_order(
    train: Sequence[float],
    max_lag: int,
    acf_lag: int,
    pacf_lag: int,
    validation_tolerance: float,
) -> tuple[int, list[dict[str, float]]]:
    """Select AR order using ACF/PACF candidates and validation RMSE.

    A small validation tail from the training period is used to avoid blindly choosing
    the highest significant correlation lag. If several lags are close, the simpler
    lag order is selected.
    """

    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    cap = min(max_lag, max(1, len(train) // 4), max(1, len(train) - 2))
    candidates = sorted({1, min(acf_lag, cap), min(pacf_lag, cap), *range(1, cap + 1)})
    validation_size = max(1, min(len(train) // 5, len(train) - cap - 1))
    fit_values = train[:-validation_size]
    validation_values = train[-validation_size:]

    scores = []
    for lag in candidates:
        if len(fit_values) <= lag:
            continue
        model = fit_ar_model(fit_values, lag)
        validation_forecast = forecast_ar(fit_values, model, len(validation_values))
        in_sample = one_step_predictions(fit_values, model)
        residuals = [actual - predicted for actual, predicted in zip(fit_values[lag:], in_sample)]
        residual_variance = sum(error**2 for error in residuals) / max(1, len(residuals))
        # AIC approximation keeps the statistical-model-selection signal visible.
        aic = len(residuals) * math.log(max(residual_variance, 1e-12)) + 2 * (lag + 1)
        scores.append(
            {
                "lag": float(lag),
                "validation_rmse": rmse(validation_values, validation_forecast),
                "in_sample_aic": aic,
            }
        )

    if not scores:
        raise ValueError("No valid lag candidates could be estimated.")
    best_rmse = min(score["validation_rmse"] for score in scores)
    close_scores = [score for score in scores if score["validation_rmse"] <= best_rmse * (1 + validation_tolerance)]
    selected = min(close_scores, key=lambda score: (score["lag"], score["in_sample_aic"]))
    return int(selected["lag"]), scores


def trading_signal(forecast: float, threshold: float) -> str:
    """Convert a forecast into a simple directional trading research signal."""

    if forecast > threshold:
        return "long"
    if forecast < -threshold:
        return "short"
    return "flat"


def svg_line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, Sequence[float]]],
    width: int = 1000,
    height: int = 420,
) -> None:
    """Write a lightweight SVG line chart without plotting dependencies."""

    margin = 50
    all_values = [value for _, values in series for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1
    max_length = max(len(values) for _, values in series)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    def point(index: int, value: float, length: int) -> tuple[float, float]:
        x_ratio = index / max(1, length - 1)
        y_ratio = (value - y_min) / (y_max - y_min)
        x = margin + x_ratio * (width - 2 * margin)
        y = height - margin - y_ratio * (height - 2 * margin)
        return x, y

    lines = []
    legend = []
    for series_index, (label, values) in enumerate(series):
        color = colors[series_index % len(colors)]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(i, value, len(values)) for i, value in enumerate(values)))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')
        legend_y = 25 + series_index * 18
        legend.append(f'<text x="{margin}" y="{legend_y}" fill="{color}" font-size="14">{label}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />
  <text x="{margin}" y="{height - 12}" font-size="12" font-family="Arial">0</text>
  <text x="{width - margin}" y="{height - 12}" text-anchor="end" font-size="12" font-family="Arial">{max_length - 1}</text>
  <text x="10" y="{margin}" font-size="12" font-family="Arial">{y_max:.4g}</text>
  <text x="10" y="{height - margin}" font-size="12" font-family="Arial">{y_min:.4g}</text>
  {''.join(legend)}
  {''.join(lines)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def svg_bar_chart(path: Path, title: str, values: Sequence[float], width: int = 1000, height: int = 420) -> None:
    """Write an SVG bar chart for ACF/PACF diagnostics."""

    margin = 50
    max_abs = max(1.0, max(abs(value) for value in values)) if values else 1.0
    zero_y = height / 2
    bar_width = (width - 2 * margin) / max(1, len(values))
    bars = []
    for index, value in enumerate(values):
        x = margin + index * bar_width
        bar_height = abs(value) / max_abs * (height / 2 - margin)
        y = zero_y - bar_height if value >= 0 else zero_y
        color = "#2563eb" if value >= 0 else "#dc2626"
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.8:.2f}" height="{bar_height:.2f}" fill="{color}" />')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>
  <line x1="{margin}" y1="{zero_y}" x2="{width - margin}" y2="{zero_y}" stroke="#333" />
  <text x="10" y="{margin}" font-size="12" font-family="Arial">+{max_abs:.2f}</text>
  <text x="10" y="{height - margin}" font-size="12" font-family="Arial">-{max_abs:.2f}</text>
  {''.join(bars)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_forecasts(
    path: Path,
    test_dates: Sequence[str],
    actual: Sequence[float],
    forecasts: Sequence[float],
    threshold: float,
) -> None:
    """Write actuals, forecasts, errors, and trading research signals to CSV."""

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["date", "actual", "forecast", "error", "signal"])
        writer.writeheader()
        for date, observed, forecast in zip(test_dates, actual, forecasts):
            writer.writerow(
                {
                    "date": date,
                    "actual": f"{observed:.10f}",
                    "forecast": f"{forecast:.10f}",
                    "error": f"{observed - forecast:.10f}",
                    "signal": trading_signal(forecast, threshold),
                }
            )


def describe_outliers(values: Sequence[float]) -> dict[str, float | int]:
    """Return simple z-score outlier diagnostics."""

    if len(values) < 2:
        return {"count": 0, "threshold_z": 3.0}
    center = mean(values)
    std_dev = statistics.stdev(values)
    if std_dev == 0:
        return {"count": 0, "threshold_z": 3.0}
    return {"count": sum(abs((value - center) / std_dev) > 3 for value in values), "threshold_z": 3.0}


def run_pipeline(config: ModelConfig) -> ModelResults:
    """Execute the complete autoregressive modeling workflow."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observations(config)
    dates = [observation.date for observation in observations]
    cleaned_values, missing_count = interpolate_missing([observation.value for observation in observations])
    prepared_values, diff_order, score_before, score_after = prepare_series(cleaned_values, config.difference)
    prepared_dates = dates[diff_order:]

    max_diagnostic_lag = min(config.max_lag, max(1, len(prepared_values) // 3), len(prepared_values) - 2)
    acf = acf_values(prepared_values, max_diagnostic_lag)
    pacf = pacf_values(prepared_values, max_diagnostic_lag)
    acf_lag = suggested_lag_from_correlation(acf, len(prepared_values))
    pacf_lag = suggested_lag_from_correlation(pacf, len(prepared_values))

    train, test = split_train_test(prepared_values, config.test_size)
    split_index = len(train)
    lag_order, lag_scores = select_lag_order(
        train,
        config.max_lag,
        acf_lag,
        pacf_lag,
        config.validation_lag_tolerance,
    )
    model = fit_ar_model(train, lag_order)
    forecasts = forecast_ar(train, model, len(test))

    forecast_path = config.output_dir / "forecasts.csv"
    metrics_path = config.output_dir / "metrics.json"
    write_forecasts(forecast_path, prepared_dates[split_index:], test, forecasts, config.signal_threshold)

    svg_line_chart(config.output_dir / "01_time_series.svg", "Historical target time series", [(config.target_column, cleaned_values)])
    svg_bar_chart(config.output_dir / "02_acf.svg", "Autocorrelation function", acf)
    svg_bar_chart(config.output_dir / "03_pacf.svg", "Partial autocorrelation function", pacf)
    svg_line_chart(config.output_dir / "04_forecast_vs_actual.svg", "AR forecast vs actual", [("actual", test), ("forecast", forecasts)])

    results = ModelResults(
        lag_order=lag_order,
        differencing_order=diff_order,
        train_observations=len(train),
        test_observations=len(test),
        mae=mae(test, forecasts),
        rmse=rmse(test, forecasts),
        acf_suggested_lag=acf_lag,
        pacf_suggested_lag=pacf_lag,
        stationarity_score_before=score_before,
        stationarity_score_after=score_after,
        forecast_path=str(forecast_path),
        metrics_path=str(metrics_path),
    )

    report = {
        "model_specification": {
            "type": "Autoregressive AR(p)",
            "lag_order_p": lag_order,
            "intercept": model.intercept,
            "coefficients": model.coefficients,
            "differencing_order": diff_order,
            "target_column": config.target_column,
        },
        "validation_results": asdict(results),
        "data_diagnostics": {
            "observations_loaded": len(observations),
            "missing_values_filled": missing_count,
            "outliers": describe_outliers(cleaned_values),
            "stationarity_score_before": score_before,
            "stationarity_score_after": score_after,
        },
        "lag_diagnostics": {
            "acf": acf,
            "pacf": pacf,
            "acf_suggested_lag": acf_lag,
            "pacf_suggested_lag": pacf_lag,
            "candidate_scores": lag_scores,
        },
        "artifacts": {
            "time_series_plot": str(config.output_dir / "01_time_series.svg"),
            "acf_plot": str(config.output_dir / "02_acf.svg"),
            "pacf_plot": str(config.output_dir / "03_pacf.svg"),
            "forecast_plot": str(config.output_dir / "04_forecast_vs_actual.svg"),
            "forecasts": str(forecast_path),
        },
        "assumptions_and_limitations": [
            "Input rows represent a consistently sampled historical time series.",
            "Missing target values are filled with linear interpolation and edge fills.",
            "Stationarity is approximated with a trend-correlation diagnostic; use formal tests when third-party packages are available.",
            "Forecast validation uses a chronological holdout and does not include transaction costs or slippage.",
            "Signals are educational research outputs and are not financial advice.",
        ],
        "refinement_ideas": [
            "Model returns instead of prices if price levels remain trend-dominated.",
            "Try exogenous predictors such as volume, index returns, macro factors, or sentiment.",
            "Use walk-forward validation before paper trading or production use.",
            "Compare against ARMA, ARIMA, SARIMAX, and simple baseline forecasts.",
        ],
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return results


def parse_args() -> ModelConfig:
    """Parse command-line options into a model configuration."""

    parser = argparse.ArgumentParser(description="Build an autoregressive trading model from CSV data.")
    parser.add_argument("--csv", dest="csv_path", required=True, type=Path, help="Path to historical CSV data.")
    parser.add_argument("--date-column", default="Date", help="Name of the datetime column in the CSV.")
    parser.add_argument("--target-column", default="Close", help="Name of the price/return column to model.")
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path, help="Directory for plots and reports.")
    parser.add_argument("--max-lag", default=20, type=int, help="Maximum AR lag order to test.")
    parser.add_argument("--test-size", default=0.2, type=float, help="Fraction of observations reserved for testing.")
    parser.add_argument("--difference", choices=["auto", "none", "first"], default="auto", help="Stationarity differencing mode.")
    parser.add_argument("--signal-threshold", default=0.0, type=float, help="Forecast threshold for long/short/flat signals.")
    parser.add_argument(
        "--validation-lag-tolerance",
        default=0.05,
        type=float,
        help="Prefer simpler lags whose validation RMSE is within this fraction of the best lag.",
    )
    args = parser.parse_args()
    return ModelConfig(**vars(args))


def main() -> None:
    """Run the CLI and print validation results."""

    results = run_pipeline(parse_args())
    print(json.dumps(asdict(results), indent=2))


if __name__ == "__main__":
    main()
