from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.validation import PortfolioError, normalize_ticker, validate_date_range, validate_finite_frame


def fetch_prices(tickers: list[str], start: str, end: str, min_rows: int = 60) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance and return a clean price frame."""
    validate_date_range(start, end)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PortfolioError(
            "yfinance is not installed. Install the packages in requirements.txt, then run again."
        ) from exc

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise PortfolioError(
            "Data error: Yahoo Finance returned no price data. Check the stock names, "
            "resolved tickers, and date range."
        )

    prices = extract_adjusted_close(raw, tickers)
    return clean_price_data(prices, tickers=tickers, min_rows=min_rows)


def get_benchmark_returns(benchmark: str, start: str, end: str) -> pd.Series:
    """Fetch benchmark prices and return daily simple returns named benchmark."""
    prices = fetch_prices([normalize_ticker(benchmark)], start, end)
    returns = calculate_simple_returns(prices)
    return returns.iloc[:, 0].rename("benchmark")


def extract_adjusted_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from a yfinance download result."""
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            raise PortfolioError("Downloaded data did not include close prices.")
    else:
        column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[column]].rename(columns={column: tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    return prices.reindex(columns=tickers)


def clean_price_data(
    prices: pd.DataFrame,
    tickers: Iterable[str] | None = None,
    min_rows: int = 60,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Clean price data with explicit ticker dropping and missing-value handling."""
    clean = validate_finite_frame(prices, "Price data").copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.dropna(axis=1, how="all")
    if fill_missing:
        clean = clean.ffill().bfill()
    clean = clean.dropna(axis=0, how="any")

    if tickers is not None:
        requested = list(tickers)
        missing = sorted(set(requested) - set(clean.columns))
        if missing:
            raise PortfolioError(
                "Data error: no usable Yahoo Finance price data for "
                f"{', '.join(missing)}. Check whether these names resolved to the right tickers."
            )
        clean = clean.reindex(columns=requested)

    if len(clean) < min_rows:
        raise PortfolioError(f"Data error: need at least {min_rows} trading days of data for a useful estimate.")
    return clean


def calculate_simple_returns(prices: pd.DataFrame, min_assets: int = 1) -> pd.DataFrame:
    """Calculate daily simple returns using the dashboard's single source of truth."""
    clean_prices = validate_finite_frame(prices, "Price data")
    returns = clean_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if returns.empty:
        raise PortfolioError("Could not compute returns from the available prices.")
    if returns.shape[1] < min_assets:
        raise PortfolioError(f"Need at least {min_assets} assets with usable return history.")
    return returns


def calculate_log_returns(prices: pd.DataFrame, min_assets: int = 1) -> pd.DataFrame:
    """Calculate log returns for diagnostics while portfolio math remains simple-return based."""
    clean_prices = validate_finite_frame(prices, "Price data")
    returns = np.log(clean_prices / clean_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if returns.empty:
        raise PortfolioError("Could not compute log returns from the available prices.")
    if returns.shape[1] < min_assets:
        raise PortfolioError(f"Need at least {min_assets} assets with usable return history.")
    return returns


def align_return_frames(*frames: pd.DataFrame | pd.Series) -> list[pd.DataFrame | pd.Series]:
    """Align return objects by the intersection of dates and drop rows with missing values."""
    if not frames:
        return []
    prepared = []
    for frame in frames:
        if isinstance(frame, pd.Series):
            prepared.append(frame.sort_index())
        else:
            prepared.append(frame.sort_index())
    common_index = prepared[0].index
    for frame in prepared[1:]:
        common_index = common_index.intersection(frame.index)
    if common_index.empty:
        raise PortfolioError("No overlapping dates were found after aligning return series.")

    aligned = []
    for frame in prepared:
        current = frame.loc[common_index]
        if isinstance(current, pd.Series):
            aligned.append(current.replace([np.inf, -np.inf], np.nan).dropna())
        else:
            aligned.append(current.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any"))

    common_index = aligned[0].index
    for frame in aligned[1:]:
        common_index = common_index.intersection(frame.index)
    return [frame.loc[common_index] for frame in aligned]


def align_strategy_and_benchmark_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Align realized strategy returns and benchmark returns by intersection of dates."""
    aligned_strategy, aligned_benchmark = align_return_frames(
        strategy_returns.rename("Portfolio"),
        benchmark_returns.rename("Benchmark"),
    )
    frame = pd.concat([aligned_strategy, aligned_benchmark], axis=1, join="inner").dropna()
    if frame.empty:
        raise PortfolioError("No overlapping strategy and benchmark returns are available.")
    return frame


def data_quality_summary(prices: pd.DataFrame, requested_tickers: Iterable[str] | None = None) -> dict[str, object]:
    """Return structured data quality diagnostics for debug mode."""
    requested = list(requested_tickers or prices.columns)
    dropped = sorted(set(requested) - set(prices.columns))
    rows_per_ticker = prices.notna().sum().astype(int).to_dict()
    missing_per_ticker = prices.isna().sum().astype(int).to_dict()
    first_valid = {column: _date_or_none(prices[column].first_valid_index()) for column in prices.columns}
    last_valid = {column: _date_or_none(prices[column].last_valid_index()) for column in prices.columns}
    return {
        "rows_per_ticker": rows_per_ticker,
        "missing_values_per_ticker": missing_per_ticker,
        "dropped_tickers": dropped,
        "first_valid_date": first_valid,
        "last_valid_date": last_valid,
        "shape": tuple(int(value) for value in prices.shape),
    }


def _date_or_none(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()

