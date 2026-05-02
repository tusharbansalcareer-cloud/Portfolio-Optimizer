from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SLIPPAGE,
    DEFAULT_TRANSACTION_COST,
    OPTIMIZATION_STRATEGIES,
    STRATEGY_CUSTOM_MANUAL,
    STRATEGY_EQUAL_WEIGHT,
)
from core.data_utils import (
    align_return_frames,
    calculate_simple_returns,
    data_quality_summary,
    fetch_prices,
)
from core.metrics import (
    drawdown_series,
    equity_curve,
    evaluate_alpha,
    performance_metrics,
    rolling_sharpe,
    total_return,
)
from core.validation import (
    PortfolioError,
    describe_ticker_resolution,
    normalize_strategy_name,
    normalize_ticker,
    normalize_tickers,
    validate_custom_weights,
    validate_date_range,
    validate_lookback_period,
    validate_rebalance_frequency,
    validate_weight_bounds,
)
from core.optimizer import OptimizationResult, PortfolioOptimizer


@dataclass(frozen=True)
class BacktestResult:
    """Streamlit-ready output from a walk-forward backtest."""

    strategy: str
    tickers: list[str]
    benchmark_ticker: str
    resolution: str
    strategy_returns: pd.Series
    gross_returns: pd.Series
    benchmark_returns: pd.Series
    equal_weight_returns: pd.Series
    comparison_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    drawdown: pd.DataFrame
    rolling_sharpe: pd.DataFrame
    weights_history: pd.DataFrame
    rebalance_table: pd.DataFrame
    cost_table: pd.DataFrame
    metrics_table: pd.DataFrame
    comparison_table: pd.DataFrame
    regression_table: pd.DataFrame
    fallback_count: int
    fallback_dates: list[str]
    debug_info: dict[str, object]


@dataclass(frozen=True)
class SingleStrategyBacktest:
    """Internal payload for one walk-forward strategy run."""

    strategy: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights_history: pd.DataFrame
    rebalance_table: pd.DataFrame
    cost_table: pd.DataFrame
    rebalance_logs: list[dict[str, object]]
    cost_logs: list[dict[str, object]]
    fallback_dates: list[str]


class TransactionCostModel:
    """Calculates turnover, transaction cost, slippage, and rebalance-day return impact."""

    def __init__(self, transaction_cost: float = DEFAULT_TRANSACTION_COST, slippage: float = DEFAULT_SLIPPAGE) -> None:
        self.transaction_cost = max(float(transaction_cost), 0.0)
        self.slippage = max(float(slippage), 0.0)

    @property
    def total_rate(self) -> float:
        """Combined cost rate charged per unit of turnover."""
        return self.transaction_cost + self.slippage

    def calculate_turnover(self, new_weights: pd.Series, previous_weights: pd.Series) -> float:
        """Calculate one-way turnover using 0.5 * sum(abs(new - old))."""
        aligned = pd.concat(
            [new_weights.rename("new"), previous_weights.rename("previous")],
            axis=1,
        ).fillna(0.0)
        return float(0.5 * (aligned["new"] - aligned["previous"]).abs().sum())

    def cost_breakdown(self, turnover: float) -> dict[str, float]:
        """Split turnover cost into transaction-cost and slippage components."""
        turnover = max(float(turnover), 0.0)
        return {
            "turnover": turnover,
            "transaction_cost": turnover * self.transaction_cost,
            "slippage_cost": turnover * self.slippage,
            "total_cost": turnover * self.total_rate,
        }

    def apply_rebalance_cost(self, returns: pd.Series, total_cost: float) -> pd.Series:
        """Apply cost only on the first realized out-of-sample day after a rebalance."""
        adjusted = returns.copy()
        if not adjusted.empty and total_cost > 0:
            adjusted.iloc[0] = float(adjusted.iloc[0]) - float(total_cost)
        return adjusted


class BacktestOptimizer:
    """Generates rebalance weights with explicit equal-weight fallback for failed periods."""

    def __init__(self, optimizer: PortfolioOptimizer | None = None) -> None:
        self.optimizer = optimizer or PortfolioOptimizer()

    def generate_weights(
        self,
        strategy: str,
        train_returns: pd.DataFrame,
        risk_free_rate: float,
        max_weight: float,
        custom_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        """Generate strategy weights; if optimization fails, use equal weight for one period."""
        try:
            return self.optimizer.optimize_strategy_weights(
                strategy,
                train_returns,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                custom_weights=custom_weights,
                allow_equal_weight_fallback=False,
            )
        except Exception as exc:
            fallback = self.optimizer.optimize_equal_weight(train_returns, risk_free_rate, max_weight)
            return OptimizationResult(
                weights=fallback.weights,
                stats=fallback.stats,
                optimizer_success=False,
                fallback_used=True,
                optimizer_name="Equal Weight fallback after optimizer failure",
                message=f"Optimizer failed; equal-weight fallback used for this rebalance only. Cause: {exc}",
            )


class WalkForwardBacktester:
    """Runs strict walk-forward backtesting with no look-ahead bias."""

    def __init__(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        rebalance_frequency: str = "Monthly",
        transaction_cost_model: TransactionCostModel | None = None,
        optimizer: BacktestOptimizer | None = None,
        debug_log_edges: int = 5,
    ) -> None:
        self.lookback_days = int(lookback_days)
        self.rebalance_frequency = validate_rebalance_frequency(rebalance_frequency)
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.optimizer = optimizer or BacktestOptimizer()
        self.debug_log_edges = int(debug_log_edges)

    def run(
        self,
        strategy: str,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        tickers: list[str],
        benchmark_ticker: str,
        resolution: str,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        custom_weights: pd.Series | None = None,
        comparison_mode: bool = False,
        debug_info: dict[str, object] | None = None,
    ) -> BacktestResult:
        """Run the selected strategy and optional comparison strategies on aligned returns."""
        strategy = normalize_strategy_name(strategy)
        validate_lookback_period(self.lookback_days, len(asset_returns))
        validate_weight_bounds(asset_returns.shape[1], max_weight)
        debug_info = debug_info or _empty_debug_info()

        main = self._run_single_strategy(
            strategy,
            asset_returns,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            custom_weights=custom_weights,
        )

        comparison_series = {
            strategy: main.net_returns,
            "Benchmark": benchmark_returns,
            "Equal Weight Basket": self._equal_weight_basket_returns(asset_returns),
        }

        if comparison_mode:
            for candidate in OPTIMIZATION_STRATEGIES:
                candidate = normalize_strategy_name(candidate)
                if candidate == strategy:
                    continue
                try:
                    comparison_run = self._run_single_strategy(
                        candidate,
                        asset_returns,
                        risk_free_rate=risk_free_rate,
                        max_weight=max_weight,
                    )
                    comparison_series[candidate] = comparison_run.net_returns
                except Exception as exc:
                    debug_info["warnings"].append(f"Comparison strategy '{candidate}' failed: {exc}")

            if custom_weights is not None and strategy != STRATEGY_CUSTOM_MANUAL:
                try:
                    comparison_run = self._run_single_strategy(
                        STRATEGY_CUSTOM_MANUAL,
                        asset_returns,
                        risk_free_rate=risk_free_rate,
                        max_weight=max_weight,
                        custom_weights=custom_weights,
                    )
                    comparison_series[STRATEGY_CUSTOM_MANUAL] = comparison_run.net_returns
                except Exception as exc:
                    debug_info["warnings"].append(f"Custom comparison strategy failed: {exc}")

        debug_info["rebalance_logs"] = _cap_logs(main.rebalance_logs, self.debug_log_edges)
        debug_info["cost_logs"] = _cap_logs(main.cost_logs, self.debug_log_edges)
        debug_info["return_checks"] = _return_checks(main.net_returns)

        report = BacktestReport(
            strategy=strategy,
            tickers=tickers,
            benchmark_ticker=benchmark_ticker,
            resolution=resolution,
            main=main,
            comparison_series=comparison_series,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            debug_info=debug_info,
        )
        return report.build()

    def _run_single_strategy(
        self,
        strategy: str,
        asset_returns: pd.DataFrame,
        risk_free_rate: float,
        max_weight: float,
        custom_weights: pd.Series | None = None,
    ) -> SingleStrategyBacktest:
        """Run one walk-forward strategy using only pre-rebalance training data."""
        rebalance_dates = self._rebalance_dates(asset_returns.index)
        if not rebalance_dates:
            raise PortfolioError("No valid rebalance dates were found after applying the lookback window.")

        old_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)
        net_chunks: list[pd.Series] = []
        gross_chunks: list[pd.Series] = []
        weight_rows: list[dict[str, object]] = []
        rebalance_rows: list[dict[str, object]] = []
        cost_rows: list[dict[str, object]] = []
        rebalance_logs: list[dict[str, object]] = []
        cost_logs: list[dict[str, object]] = []
        fallback_dates: list[str] = []

        for index, rebalance_date in enumerate(rebalance_dates):
            # Look-ahead prevention: training returns end strictly before rebalance_date.
            train_returns = asset_returns.loc[asset_returns.index < rebalance_date].tail(self.lookback_days)
            if len(train_returns) < self.lookback_days:
                continue

            next_rebalance = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else None
            if next_rebalance is None:
                period_returns = asset_returns.loc[asset_returns.index >= rebalance_date]
            else:
                period_returns = asset_returns.loc[
                    (asset_returns.index >= rebalance_date) & (asset_returns.index < next_rebalance)
                ]
            if period_returns.empty:
                continue

            optimization = self.optimizer.generate_weights(
                strategy,
                train_returns,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                custom_weights=custom_weights,
            )
            turnover = self.transaction_cost_model.calculate_turnover(optimization.weights, old_weights)
            costs = self.transaction_cost_model.cost_breakdown(turnover)
            gross = optimization.weights.reindex(period_returns.columns).fillna(0.0).dot(period_returns.T)
            gross = pd.Series(gross.to_numpy(dtype=float), index=period_returns.index, name=strategy)
            net = self.transaction_cost_model.apply_rebalance_cost(gross, costs["total_cost"])
            net.name = strategy

            rebalance_label = pd.Timestamp(rebalance_date).date().isoformat()
            if optimization.fallback_used:
                fallback_dates.append(rebalance_label)
            if turnover > 0.75:
                # High turnover is not fatal, but it is important for production diagnostics.
                cost_logs.append({"date": rebalance_label, "warning": "High turnover", "turnover": turnover})

            weight_row = {"Date": pd.Timestamp(rebalance_date)}
            weight_row.update({ticker: float(optimization.weights.get(ticker, 0.0)) for ticker in asset_returns.columns})
            weight_rows.append(weight_row)

            rebalance_row = {
                "Date": pd.Timestamp(rebalance_date),
                "Training Start": train_returns.index[0],
                "Training End": train_returns.index[-1],
                "Observations": int(len(train_returns)),
                "Optimizer": optimization.optimizer_name,
                "Optimizer Success": optimization.optimizer_success,
                "Fallback Used": optimization.fallback_used,
                "Message": optimization.message,
            }
            rebalance_rows.append(rebalance_row)

            cost_row = {
                "Date": pd.Timestamp(rebalance_date),
                "Turnover": costs["turnover"],
                "Transaction Cost": costs["transaction_cost"],
                "Slippage Cost": costs["slippage_cost"],
                "Total Cost": costs["total_cost"],
            }
            cost_rows.append(cost_row)

            rebalance_logs.append(
                {
                    "date": rebalance_label,
                    "training_window_start": train_returns.index[0].date().isoformat(),
                    "training_window_end": train_returns.index[-1].date().isoformat(),
                    "training_observations": int(len(train_returns)),
                    "optimizer_success": optimization.optimizer_success,
                    "fallback_used": optimization.fallback_used,
                    "optimizer_message": optimization.message,
                    "weights": {ticker: float(value) for ticker, value in optimization.weights.items()},
                }
            )
            cost_logs.append(
                {
                    "date": rebalance_label,
                    "previous_weights": {ticker: float(old_weights.get(ticker, 0.0)) for ticker in asset_returns.columns},
                    "new_weights": {ticker: float(optimization.weights.get(ticker, 0.0)) for ticker in asset_returns.columns},
                    "turnover": costs["turnover"],
                    "transaction_cost": costs["transaction_cost"],
                    "slippage_cost": costs["slippage_cost"],
                    "net_cost_applied": costs["total_cost"],
                }
            )

            gross_chunks.append(gross)
            net_chunks.append(net)
            old_weights = optimization.weights.reindex(asset_returns.columns).fillna(0.0)

        if not net_chunks:
            raise PortfolioError("Backtest produced no out-of-sample return periods.")

        return SingleStrategyBacktest(
            strategy=strategy,
            net_returns=pd.concat(net_chunks).sort_index(),
            gross_returns=pd.concat(gross_chunks).sort_index(),
            weights_history=pd.DataFrame(weight_rows).set_index("Date").sort_index(),
            rebalance_table=pd.DataFrame(rebalance_rows),
            cost_table=pd.DataFrame(cost_rows),
            rebalance_logs=rebalance_logs,
            cost_logs=cost_logs,
            fallback_dates=fallback_dates,
        )

    def _rebalance_dates(self, index: pd.Index) -> list[pd.Timestamp]:
        """Use first actual trading date in each rebalance bucket after lookback."""
        eligible_index = pd.DatetimeIndex(index).sort_values()
        if len(eligible_index) <= self.lookback_days:
            return []
        eligible_index = eligible_index[self.lookback_days :]
        alias = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}[self.rebalance_frequency]
        dates = []
        for _, group in pd.Series(eligible_index, index=eligible_index).groupby(eligible_index.to_period(alias)):
            if not group.empty:
                dates.append(pd.Timestamp(group.iloc[0]))
        return dates

    @staticmethod
    def _equal_weight_basket_returns(asset_returns: pd.DataFrame) -> pd.Series:
        """Return the selected-universe equal-weight basket for baseline comparison."""
        weights = pd.Series(1.0 / asset_returns.shape[1], index=asset_returns.columns)
        return asset_returns.mul(weights, axis=1).sum(axis=1).rename("Equal Weight Basket")


class BacktestReport:
    """Prepares clean output objects for Streamlit."""

    def __init__(
        self,
        strategy: str,
        tickers: list[str],
        benchmark_ticker: str,
        resolution: str,
        main: SingleStrategyBacktest,
        comparison_series: dict[str, pd.Series],
        initial_capital: float,
        risk_free_rate: float,
        debug_info: dict[str, object],
    ) -> None:
        self.strategy = strategy
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.resolution = resolution
        self.main = main
        self.comparison_series = comparison_series
        self.initial_capital = float(initial_capital)
        self.risk_free_rate = float(risk_free_rate)
        self.debug_info = debug_info

    def build(self) -> BacktestResult:
        """Build display frames, metrics, warnings, and regression output."""
        comparison_returns = pd.concat(self.comparison_series.values(), axis=1, join="inner").dropna()
        comparison_returns.columns = list(self.comparison_series.keys())
        strategy_returns = comparison_returns[self.strategy].rename(self.strategy)
        benchmark_returns = comparison_returns["Benchmark"].rename("Benchmark")
        equal_weight_returns = comparison_returns["Equal Weight Basket"].rename("Equal Weight Basket")

        equity = comparison_returns.apply(lambda series: equity_curve(series, self.initial_capital))
        drawdowns = comparison_returns.apply(drawdown_series)
        rolling = comparison_returns.apply(lambda series: rolling_sharpe(series, window=63, risk_free_rate=self.risk_free_rate))

        metrics = _safe_performance_metrics(strategy_returns, benchmark_returns, self.risk_free_rate)
        benchmark_metrics = _safe_performance_metrics(benchmark_returns, None, self.risk_free_rate)
        equal_metrics = _safe_performance_metrics(equal_weight_returns, benchmark_returns, self.risk_free_rate)
        metrics_table = _metrics_rows(
            {
                self.strategy: metrics,
                "Benchmark": benchmark_metrics,
                "Equal Weight Basket": equal_metrics,
            }
        )

        comparison_table = _metrics_rows(
            {
                column: _safe_performance_metrics(
                    comparison_returns[column],
                    benchmark_returns if column != "Benchmark" else None,
                    self.risk_free_rate,
                )
                for column in comparison_returns.columns
            }
        )
        regression_table = _regression_table(strategy_returns, benchmark_returns)

        self.debug_info["metric_checks"] = metrics
        _append_warning_checks(self.debug_info, metrics, strategy_returns, self.main.cost_table, self.main.fallback_dates)

        return BacktestResult(
            strategy=self.strategy,
            tickers=self.tickers,
            benchmark_ticker=self.benchmark_ticker,
            resolution=self.resolution,
            strategy_returns=strategy_returns,
            gross_returns=self.main.gross_returns.reindex(strategy_returns.index).dropna(),
            benchmark_returns=benchmark_returns,
            equal_weight_returns=equal_weight_returns,
            comparison_returns=comparison_returns,
            equity_curve=equity,
            drawdown=drawdowns,
            rolling_sharpe=rolling,
            weights_history=self.main.weights_history,
            rebalance_table=self.main.rebalance_table,
            cost_table=self.main.cost_table,
            metrics_table=metrics_table,
            comparison_table=comparison_table,
            regression_table=regression_table,
            fallback_count=len(self.main.fallback_dates),
            fallback_dates=self.main.fallback_dates,
            debug_info=self.debug_info,
        )


def run_walk_forward_backtest(
    tickers: str | Iterable[str],
    benchmark: str = DEFAULT_BENCHMARK,
    start: str = "2020-01-01",
    end: str = "2026-04-30",
    strategy: str = "Mean-Variance / Max Sharpe",
    rebalance_frequency: str = "Monthly",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    slippage: float = DEFAULT_SLIPPAGE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    custom_weights_df: pd.DataFrame | None = None,
    normalize_custom_weights: bool = True,
    comparison_mode: bool = False,
    debug: bool = False,
) -> BacktestResult:
    """Convenience API used by Streamlit to run a complete walk-forward backtest."""
    start_ts, end_ts = validate_date_range(start, end)
    resolved_tickers = normalize_tickers(tickers)
    benchmark_ticker = normalize_ticker(benchmark)
    strategy = normalize_strategy_name(strategy)
    validate_rebalance_frequency(rebalance_frequency)
    validate_weight_bounds(len(resolved_tickers), max_weight)
    resolution = describe_ticker_resolution(tickers, resolved_tickers)

    debug_info = _empty_debug_info()
    debug_info["inputs"] = {
        "selected_tickers": resolved_tickers,
        "date_range": {"start": start_ts.date().isoformat(), "end": end_ts.date().isoformat()},
        "lookback_window": int(lookback_days),
        "rebalance_frequency": rebalance_frequency,
        "optimization_strategy": strategy,
        "benchmark_ticker": benchmark_ticker,
        "transaction_cost": float(transaction_cost),
        "slippage": float(slippage),
        "risk_free_rate": float(risk_free_rate),
        "initial_capital": float(initial_capital),
        "comparison_mode": bool(comparison_mode),
        "debug_enabled": bool(debug),
    }

    prices = fetch_prices(resolved_tickers, start_ts.date().isoformat(), end_ts.date().isoformat(), min_rows=lookback_days + 2)
    benchmark_prices = fetch_prices([benchmark_ticker], start_ts.date().isoformat(), end_ts.date().isoformat(), min_rows=60)
    debug_info["data_quality"] = {
        "assets": data_quality_summary(prices, resolved_tickers),
        "benchmark": data_quality_summary(benchmark_prices, [benchmark_ticker]),
    }

    asset_returns = calculate_simple_returns(prices, min_assets=2)
    benchmark_returns = calculate_simple_returns(benchmark_prices, min_assets=1).iloc[:, 0].rename("Benchmark")
    asset_returns, benchmark_returns = align_return_frames(asset_returns, benchmark_returns)
    benchmark_returns = benchmark_returns.rename("Benchmark")
    debug_info["data_quality"]["final_aligned_shape"] = tuple(int(value) for value in asset_returns.shape)
    debug_info["data_quality"]["benchmark_alignment_status"] = "aligned"

    if len(asset_returns) < 252:
        debug_info["warnings"].append("Backtest period is shorter than one trading year.")

    custom_weights = None
    if custom_weights_df is not None and not custom_weights_df.empty:
        custom_weights = validate_custom_weights(custom_weights_df, resolved_tickers, normalize=normalize_custom_weights)
    elif strategy == STRATEGY_CUSTOM_MANUAL:
        raise PortfolioError("Custom Manual Allocation requires custom weights.")

    cost_model = TransactionCostModel(transaction_cost=transaction_cost, slippage=slippage)
    backtester = WalkForwardBacktester(
        lookback_days=lookback_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_model=cost_model,
    )
    return backtester.run(
        strategy=strategy,
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        tickers=resolved_tickers,
        benchmark_ticker=benchmark_ticker,
        resolution=resolution,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        custom_weights=custom_weights,
        comparison_mode=comparison_mode,
        debug_info=debug_info,
    )


def _empty_debug_info() -> dict[str, object]:
    return {
        "inputs": {},
        "data_quality": {},
        "rebalance_logs": [],
        "cost_logs": [],
        "return_checks": {},
        "metric_checks": {},
        "warnings": [],
    }


def _cap_logs(logs: list[dict[str, object]], edge_count: int) -> list[dict[str, object]]:
    if len(logs) <= edge_count * 2:
        return logs
    return logs[:edge_count] + [{"omitted_middle_logs": len(logs) - edge_count * 2}] + logs[-edge_count:]


def _return_checks(returns: pd.Series) -> dict[str, object]:
    values = returns.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    checks = {
        "first_returns": {str(key.date()): float(value) for key, value in returns.head().items()},
        "last_returns": {str(key.date()): float(value) for key, value in returns.tail().items()},
        "nan_count": int(returns.isna().sum()),
        "infinite_count": int(np.isinf(values).sum()),
        "min_daily_return": float(np.min(finite)) if len(finite) else np.nan,
        "max_daily_return": float(np.max(finite)) if len(finite) else np.nan,
    }
    extreme = returns[returns.abs() > 0.2]
    if not extreme.empty:
        checks["extreme_daily_return_dates"] = [date.date().isoformat() for date in extreme.index]
    return checks


def _safe_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free_rate: float,
) -> dict[str, float]:
    try:
        return performance_metrics(returns, benchmark_returns, risk_free_rate=risk_free_rate)
    except Exception:
        return performance_metrics(returns, None, risk_free_rate=risk_free_rate)


def _metrics_rows(metrics_by_series: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for label, metrics in metrics_by_series.items():
        row = {"Series": label}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _regression_table(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    try:
        alpha = evaluate_alpha(strategy_returns, benchmark_returns, alpha_threshold=-np.inf, information_ratio_threshold=-np.inf)
        return pd.DataFrame(
            [
                {
                    "Alpha": alpha.alpha_annualized,
                    "Beta": alpha.beta,
                    "Information Ratio": alpha.information_ratio,
                    "Tracking Error": alpha.tracking_error,
                    "Alpha p-value": np.nan if alpha.alpha_p_value is None else alpha.alpha_p_value,
                    "Observations": alpha.observations,
                }
            ]
        )
    except Exception as exc:
        return pd.DataFrame([{"Alpha": np.nan, "Beta": np.nan, "Information Ratio": np.nan, "Tracking Error": np.nan, "Error": str(exc)}])


def _append_warning_checks(
    debug_info: dict[str, object],
    metrics: dict[str, float],
    strategy_returns: pd.Series,
    cost_table: pd.DataFrame,
    fallback_dates: list[str],
) -> None:
    warnings = debug_info["warnings"]
    if fallback_dates:
        warnings.append(f"Fallback triggered in {len(fallback_dates)} rebalances.")
    if strategy_returns.isna().any():
        warnings.append("NaN values found after return calculation.")
    if np.isinf(strategy_returns.to_numpy(dtype=float)).any():
        warnings.append("Infinite values found after return calculation.")
    if (strategy_returns.abs() > 0.2).any():
        warnings.append("Extreme daily return greater than 20% detected.")
    if not cost_table.empty and float(cost_table["Turnover"].max()) > 0.75:
        warnings.append("Very high turnover detected.")
    max_dd = float(metrics.get("Max Drawdown", np.nan))
    if np.isfinite(max_dd) and max_dd < -0.30:
        warnings.append("Very high drawdown detected.")
    sharpe = float(metrics.get("Sharpe Ratio", np.nan))
    if np.isfinite(sharpe) and sharpe > 5:
        warnings.append("Suspiciously high Sharpe ratio detected.")
