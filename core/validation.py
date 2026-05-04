from __future__ import annotations

from datetime import date, datetime
from difflib import get_close_matches
from typing import Iterable
import re

import numpy as np
import pandas as pd

from core.constants import (
    NAME_TO_YAHOO_TICKER,
    REBALANCE_PERIOD_ALIASES,
    STRATEGY_ALIASES,
    SUPPORTED_REBALANCE_FREQUENCIES,
)


YAHOO_TICKER_PATTERN = re.compile(r"^[A-Z0-9.&^=-]+(?:\.[A-Z]{1,4})?$")


class PortfolioError(ValueError):
    """Raised when the portfolio pipeline cannot continue with the inputs."""


def normalize_strategy_name(strategy: str) -> str:
    """Return the canonical strategy name used by both engines."""
    strategy = str(strategy).strip()
    if strategy in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[strategy]
    raise PortfolioError(f"Unknown optimization strategy: {strategy}")


def normalize_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    """Resolve raw stock names or Yahoo tickers into unique Yahoo symbols."""
    if isinstance(raw_tickers, str):
        raw_items = [item.strip() for item in raw_tickers.split(",")]
    else:
        raw_items = [str(item).strip() for item in raw_tickers]

    resolved = [_resolve_to_yahoo_ticker(item) for item in raw_items if item]
    deduped = list(dict.fromkeys(resolved))
    if len(deduped) < 2:
        raise PortfolioError(
            "Input error: enter at least two valid stocks or indexes, for example "
            "'reliance, tcs, hdfc bank' or 'nifty 50, bank nifty'."
        )
    return deduped


def normalize_ticker(raw_ticker: str) -> str:
    """Resolve a single raw name or ticker into a Yahoo Finance symbol."""
    raw_ticker = str(raw_ticker).strip()
    if not raw_ticker:
        raise PortfolioError("Input error: benchmark cannot be blank.")
    return _resolve_to_yahoo_ticker(raw_ticker)


def describe_ticker_resolution(raw_tickers: str | Iterable[str], resolved_tickers: Iterable[str]) -> str:
    """Build a compact human-readable raw-to-Yahoo ticker mapping."""
    if isinstance(raw_tickers, str):
        raw_items = [item.strip() for item in raw_tickers.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in raw_tickers if str(item).strip()]

    pairs = []
    seen = set()
    for raw, ticker in zip(raw_items, resolved_tickers):
        if ticker in seen:
            continue
        seen.add(ticker)
        pairs.append(f"{raw} -> {ticker}")
    return ", ".join(pairs)


def validate_date_range(start: str | date | datetime, end: str | date | datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a start/end date pair."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise PortfolioError("Start and end dates must be valid dates.")
    if start_ts >= end_ts:
        raise PortfolioError("Start date must be before end date.")
    return start_ts, end_ts


def validate_lookback_period(lookback_days: int, available_observations: int | None = None) -> int:
    """Validate a walk-forward lookback window in trading days."""
    lookback_days = int(lookback_days)
    if lookback_days < 30:
        raise PortfolioError("Lookback window must be at least 30 trading days.")
    if available_observations is not None and available_observations <= lookback_days:
        raise PortfolioError(
            f"Insufficient data for a {lookback_days}-day lookback. "
            f"Only {available_observations} aligned return observations are available."
        )
    return lookback_days


def validate_rebalance_frequency(rebalance_frequency: str) -> str:
    """Validate and return a canonical rebalance frequency label."""
    frequency = str(rebalance_frequency).strip().title()
    if frequency not in SUPPORTED_REBALANCE_FREQUENCIES:
        allowed = ", ".join(SUPPORTED_REBALANCE_FREQUENCIES)
        raise PortfolioError(f"Unsupported rebalance frequency '{rebalance_frequency}'. Use one of: {allowed}.")
    return frequency


def rebalance_period_alias(rebalance_frequency: str) -> str:
    """Return the pandas period alias for a validated rebalance frequency."""
    frequency = validate_rebalance_frequency(rebalance_frequency)
    if frequency not in REBALANCE_PERIOD_ALIASES:
        raise PortfolioError(f"Rebalance frequency '{frequency}' does not use a calendar period alias.")
    return REBALANCE_PERIOD_ALIASES[frequency]


def validate_weight_bounds(n_assets: int, max_weight: float) -> None:
    """Ensure long-only fully invested weights can satisfy the max-weight cap."""
    if n_assets <= 0:
        raise PortfolioError("At least one asset is required.")
    if not 0 < float(max_weight) <= 1:
        raise PortfolioError("Maximum asset weight must be between 0 and 1.")
    if float(max_weight) * n_assets < 1 - 1e-10:
        raise PortfolioError(
            "Maximum single-stock weight is too low for the number of assets. "
            f"Use at least {1 / n_assets:.2%} for {n_assets} assets."
        )


def normalize_weights(weights: pd.Series | dict[str, float], index: Iterable[str] | None = None) -> pd.Series:
    """Normalize non-negative finite weights to sum to one."""
    series = pd.Series(weights, dtype=float)
    if index is not None:
        series = series.reindex(list(index)).fillna(0.0)
    if series.empty:
        raise PortfolioError("Weights cannot be empty.")
    if not np.all(np.isfinite(series.to_numpy(dtype=float))):
        raise PortfolioError("Weights must be finite numeric values.")
    if (series < 0).any():
        invalid = ", ".join(series[series < 0].index.astype(str))
        raise PortfolioError(f"Weights cannot be negative for: {invalid}.")
    total = float(series.sum())
    if total <= 0:
        raise PortfolioError("At least one weight must be greater than zero.")
    return (series / total).astype(float)


def check_weights_sum_to_one(weights: pd.Series, tolerance: float = 1e-6) -> None:
    """Validate that a weight vector is finite, non-negative, and fully invested."""
    if weights.empty:
        raise PortfolioError("Weights cannot be empty.")
    if not np.all(np.isfinite(weights.to_numpy(dtype=float))):
        raise PortfolioError("Weights must be finite numeric values.")
    if (weights < -tolerance).any():
        raise PortfolioError("Weights cannot contain negative values.")
    if not np.isclose(float(weights.sum()), 1.0, atol=tolerance):
        raise PortfolioError(f"Weights must sum to 1. Current sum is {float(weights.sum()):.8f}.")


def validate_custom_weights(
    custom_weights_df: pd.DataFrame,
    tickers: list[str],
    normalize: bool = True,
) -> pd.Series:
    """Validate manual allocation percentages and return proportions indexed by ticker."""
    required_columns = {"Ticker", "Custom Weight %"}
    if custom_weights_df is None or custom_weights_df.empty:
        raise PortfolioError("Custom weights are enabled, but no weights were provided.")
    if not required_columns.issubset(custom_weights_df.columns):
        raise PortfolioError("Custom weights must include Ticker and Custom Weight % columns.")

    weights_df = custom_weights_df.loc[:, ["Ticker", "Custom Weight %"]].copy()
    weights_df["Ticker"] = weights_df["Ticker"].astype(str).str.strip()
    if weights_df["Ticker"].duplicated().any():
        duplicates = sorted(weights_df.loc[weights_df["Ticker"].duplicated(), "Ticker"].unique())
        raise PortfolioError(f"Custom weights contain duplicate tickers: {', '.join(duplicates)}.")

    missing = [ticker for ticker in tickers if ticker not in set(weights_df["Ticker"])]
    if missing:
        raise PortfolioError(f"Custom weights are missing allocations for: {', '.join(missing)}.")

    extra = sorted(set(weights_df["Ticker"]) - set(tickers))
    if extra:
        raise PortfolioError(f"Custom weights include tickers that are not in the portfolio: {', '.join(extra)}.")

    weights = pd.to_numeric(weights_df.set_index("Ticker")["Custom Weight %"], errors="coerce").reindex(tickers)
    if weights.isna().any():
        invalid = ", ".join(weights[weights.isna()].index)
        raise PortfolioError(f"Custom weights must be numeric for: {invalid}.")
    if (weights < 0).any():
        invalid = ", ".join(weights[weights < 0].index)
        raise PortfolioError(f"Custom weights cannot be negative for: {invalid}.")
    if not (weights > 0).any():
        raise PortfolioError("At least one custom weight must be greater than 0%.")

    total_percent = float(weights.sum())
    if normalize:
        return (weights / total_percent).astype(float)
    if not np.isclose(total_percent, 100.0, atol=1e-6):
        raise PortfolioError(
            f"Custom weights must sum to 100% when normalization is off. Current sum is {total_percent:.4f}%."
        )
    return (weights / 100.0).astype(float)


def validate_finite_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure a DataFrame contains at least one finite row and no infinite values."""
    if frame is None or frame.empty:
        raise PortfolioError(f"{label} cannot be empty.")
    clean = frame.replace([np.inf, -np.inf], np.nan)
    if clean.dropna(how="all").empty:
        raise PortfolioError(f"{label} has no finite observations.")
    return clean


def _resolve_to_yahoo_ticker(raw_value: str) -> str:
    value = raw_value.strip()
    upper_value = value.upper()
    if not value:
        raise PortfolioError("Input error: blank stock name found.")

    key = _normalize_lookup_key(value)
    if key in NAME_TO_YAHOO_TICKER:
        return NAME_TO_YAHOO_TICKER[key]

    if _looks_like_yahoo_ticker(upper_value):
        return upper_value

    close_matches = get_close_matches(key, NAME_TO_YAHOO_TICKER.keys(), n=1, cutoff=0.84)
    if close_matches:
        return NAME_TO_YAHOO_TICKER[close_matches[0]]

    candidate = re.sub(r"[^A-Z0-9&-]", "", upper_value)
    if candidate:
        return f"{candidate}.NS"

    raise PortfolioError(
        f"Input error: could not convert '{raw_value}' to a Yahoo Finance ticker. "
        "Try the company name, for example 'reliance', 'adani ports', or a direct ticker like 'RELIANCE.NS'."
    )


def _looks_like_yahoo_ticker(value: str) -> bool:
    if " " in value:
        return False
    return bool(YAHOO_TICKER_PATTERN.fullmatch(value)) and (
        value.startswith("^") or "." in value or value.endswith("=F") or len(value) <= 5
    )


def _normalize_lookup_key(value: str) -> str:
    key = value.lower().replace("&", " and ")
    key = re.sub(r"[^a-z0-9]+", " ", key)
    key = re.sub(r"\b(limited|ltd|company|co|stock|shares|share)\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()
