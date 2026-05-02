from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE_RATE,
    STRATEGY_CUSTOM_MANUAL,
    STRATEGY_EQUAL_WEIGHT,
    STRATEGY_MEAN_VARIANCE,
    STRATEGY_MINIMUM_VARIANCE,
    STRATEGY_RISK_PARITY,
)
from core.metrics import (
    PortfolioStats,
    clean_returns,
    estimate_annualized_inputs,
    portfolio_stats,
    risk_contribution_array,
)
from core.validation import (
    PortfolioError,
    check_weights_sum_to_one,
    normalize_strategy_name,
    normalize_weights,
    validate_weight_bounds,
)


@dataclass(frozen=True)
class OptimizationResult:
    """Structured optimizer output with explicit fallback metadata."""

    weights: pd.Series
    stats: PortfolioStats
    optimizer_success: bool
    fallback_used: bool
    optimizer_name: str
    message: str


class PortfolioOptimizer:
    """Own all portfolio weight generation for optimization and backtesting engines."""

    def optimize_strategy_weights(
        self,
        strategy: str,
        returns: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        l2_reg: float = 0.0,
        custom_weights: pd.Series | None = None,
        allow_equal_weight_fallback: bool = False,
    ) -> OptimizationResult:
        """Generate weights for one canonical strategy without duplicating optimizer logic."""
        strategy = normalize_strategy_name(strategy)
        try:
            if strategy == STRATEGY_MEAN_VARIANCE:
                return self.optimize_max_sharpe(returns, risk_free_rate, max_weight, l2_reg)
            if strategy == STRATEGY_MINIMUM_VARIANCE:
                return self.optimize_min_variance(returns, risk_free_rate, max_weight)
            if strategy == STRATEGY_RISK_PARITY:
                return self.optimize_risk_parity(returns, risk_free_rate, max_weight)
            if strategy == STRATEGY_EQUAL_WEIGHT:
                return self.optimize_equal_weight(returns, risk_free_rate, max_weight)
            if strategy == STRATEGY_CUSTOM_MANUAL:
                if custom_weights is None:
                    raise PortfolioError("Custom manual weights were requested but not provided.")
                return self.apply_custom_weights(returns, custom_weights, risk_free_rate)
        except Exception as exc:
            if not allow_equal_weight_fallback:
                raise
            return self._equal_weight_fallback(returns, risk_free_rate, max_weight, str(exc))
        raise PortfolioError(f"Unknown optimization strategy: {strategy}")

    def optimize_max_sharpe(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        l2_reg: float = 0.0,
    ) -> OptimizationResult:
        """Optimize long-only weights for maximum annualized Sharpe ratio."""
        mean_returns, covariance = estimate_annualized_inputs(returns)
        validate_weight_bounds(len(mean_returns), max_weight)
        mu = mean_returns.to_numpy(dtype=float)
        cov = covariance.to_numpy(dtype=float)
        l2_penalty = max(float(l2_reg), 0.0)

        def objective(weights: np.ndarray) -> float:
            expected = float(weights @ mu)
            volatility = float(np.sqrt(weights.T @ cov @ weights))
            if volatility <= 0 or not np.isfinite(volatility):
                return np.inf
            return -((expected - risk_free_rate) / volatility) + l2_penalty * float(weights @ weights)

        optimizer_name = "Max Sharpe / SciPy SLSQP"
        message = "Optimization completed with SciPy SLSQP."
        try:
            weights = _solve_constrained_weights(objective, mean_returns.index, max_weight)
        except Exception as exc:
            weights = _random_search_weights(objective, mean_returns.index, max_weight)
            optimizer_name = "Max Sharpe / NumPy random search fallback"
            message = f"SciPy failed; NumPy random search produced weights. Cause: {exc}"

        return self._result_from_weights(weights, mean_returns, covariance, risk_free_rate, optimizer_name, message)

    def optimize_min_variance(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
    ) -> OptimizationResult:
        """Optimize long-only weights for minimum variance."""
        mean_returns, covariance = estimate_annualized_inputs(returns)
        validate_weight_bounds(len(mean_returns), max_weight)
        cov = covariance.to_numpy(dtype=float)

        def objective(weights: np.ndarray) -> float:
            return float(weights.T @ cov @ weights)

        optimizer_name = "Minimum Variance / SciPy SLSQP"
        message = "Optimization completed with SciPy SLSQP."
        try:
            weights = _solve_constrained_weights(objective, mean_returns.index, max_weight)
        except Exception as exc:
            weights = _random_search_weights(objective, mean_returns.index, max_weight, maximize=False)
            optimizer_name = "Minimum Variance / NumPy random search fallback"
            message = f"SciPy failed; NumPy random search produced weights. Cause: {exc}"

        return self._result_from_weights(weights, mean_returns, covariance, risk_free_rate, optimizer_name, message)

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
    ) -> OptimizationResult:
        """Optimize long-only weights for equalized variance risk contribution."""
        mean_returns, covariance = estimate_annualized_inputs(returns)
        validate_weight_bounds(len(mean_returns), max_weight)
        cov = covariance.to_numpy(dtype=float)
        target = np.full(len(mean_returns), 1.0 / len(mean_returns))

        def objective(weights: np.ndarray) -> float:
            contribution = risk_contribution_array(weights, cov)
            return float(np.sum((contribution - target) ** 2))

        optimizer_name = "Risk Parity / SciPy SLSQP"
        message = "Optimization completed with SciPy SLSQP."
        try:
            weights = _solve_constrained_weights(objective, mean_returns.index, max_weight)
        except Exception as exc:
            weights = _multiplicative_risk_parity_weights(covariance, max_weight)
            optimizer_name = "Risk Parity / NumPy iterative fallback"
            message = f"SciPy failed; iterative risk parity produced weights. Cause: {exc}"

        return self._result_from_weights(weights, mean_returns, covariance, risk_free_rate, optimizer_name, message)

    def optimize_equal_weight(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_weight: float = DEFAULT_MAX_WEIGHT,
    ) -> OptimizationResult:
        """Return a long-only equal-weight allocation."""
        mean_returns, covariance = estimate_annualized_inputs(returns)
        validate_weight_bounds(len(mean_returns), max_weight)
        weights = pd.Series(1.0 / len(mean_returns), index=mean_returns.index, dtype=float)
        return self._result_from_weights(
            weights,
            mean_returns,
            covariance,
            risk_free_rate,
            "Equal Weight / 1/N",
            "Equal-weight allocation applied.",
        )

    def apply_custom_weights(
        self,
        returns: pd.DataFrame,
        custom_weights: pd.Series,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> OptimizationResult:
        """Normalize and apply user-supplied manual weights to the active return universe."""
        clean = clean_returns(returns)
        weights = normalize_weights(custom_weights, clean.columns)
        check_weights_sum_to_one(weights)
        mean_returns, covariance = estimate_annualized_inputs(clean)
        return self._result_from_weights(
            weights,
            mean_returns,
            covariance,
            risk_free_rate,
            "Custom Manual Allocation / user weights",
            "Validated custom manual allocation applied.",
        )

    def _equal_weight_fallback(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        max_weight: float,
        failure_message: str,
    ) -> OptimizationResult:
        """Use equal weight for one failed optimization period without persisting it."""
        result = self.optimize_equal_weight(returns, risk_free_rate, max_weight)
        stats = PortfolioStats(
            expected_return=result.stats.expected_return,
            volatility=result.stats.volatility,
            sharpe_ratio=result.stats.sharpe_ratio,
            optimizer="Equal Weight fallback after optimizer failure",
        )
        return OptimizationResult(
            weights=result.weights,
            stats=stats,
            optimizer_success=False,
            fallback_used=True,
            optimizer_name=stats.optimizer,
            message=f"Optimizer failed; equal-weight fallback used for this period only. Cause: {failure_message}",
        )

    @staticmethod
    def _result_from_weights(
        weights: pd.Series,
        mean_returns: pd.Series,
        covariance: pd.DataFrame,
        risk_free_rate: float,
        optimizer_name: str,
        message: str,
    ) -> OptimizationResult:
        """Package weights with shared portfolio statistics."""
        stats = portfolio_stats(weights, mean_returns, covariance, risk_free_rate, optimizer_name)
        return OptimizationResult(
            weights=weights,
            stats=stats,
            optimizer_success=True,
            fallback_used=False,
            optimizer_name=optimizer_name,
            message=message,
        )


def _solve_constrained_weights(objective, index: pd.Index, max_weight: float) -> pd.Series:
    """Solve a long-only fully invested weight optimization with SciPy."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise PortfolioError("SciPy is not installed. Install requirements.txt, then run again.") from exc

    n_assets = len(index)
    initial = np.full(n_assets, 1.0 / n_assets)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise PortfolioError(f"Optimization failed: {result.message}")
    return _weights_from_array(result.x, index, max_weight)


def _random_search_weights(
    objective,
    index: pd.Index,
    max_weight: float,
    maximize: bool = False,
    attempts: int = 1200,
) -> pd.Series:
    """Deterministic random-search fallback when SciPy optimization is unavailable or fails."""
    rng = np.random.default_rng(7)
    n_assets = len(index)
    best_weights: np.ndarray | None = None
    best_score = -np.inf if maximize else np.inf

    for _ in range(attempts):
        candidate = _project_to_capped_simplex(rng.dirichlet(np.ones(n_assets)), max_weight)
        score = float(objective(candidate))
        if not np.isfinite(score):
            continue
        is_better = score > best_score if maximize else score < best_score
        if is_better:
            best_score = score
            best_weights = candidate

    if best_weights is None:
        raise PortfolioError("Could not find a valid portfolio under the selected weight cap.")
    return _weights_from_array(best_weights, index, max_weight)


def _multiplicative_risk_parity_weights(covariance: pd.DataFrame, max_weight: float) -> pd.Series:
    """Iteratively solve risk parity and fall back to random search if convergence is poor."""
    cov = covariance.to_numpy(dtype=float)
    n_assets = len(covariance.index)
    target = np.full(n_assets, 1.0 / n_assets)
    weights = np.full(n_assets, 1.0 / n_assets)
    best_weights = weights.copy()
    best_error = np.inf

    for _ in range(2000):
        contribution = risk_contribution_array(weights, cov)
        error = float(np.sum((contribution - target) ** 2))
        if error < best_error:
            best_error = error
            best_weights = weights.copy()
        if error < 1e-10:
            break
        if np.any(contribution <= 0):
            break
        adjustment = np.sqrt(np.clip(target / np.clip(contribution, 1e-8, None), 0.05, 20.0))
        weights = _project_to_capped_simplex(weights * adjustment, max_weight)

    if not np.isfinite(best_error) or best_error > 5e-4:

        def objective(candidate: np.ndarray) -> float:
            contribution = risk_contribution_array(candidate, cov)
            return float(np.sum((contribution - target) ** 2))

        return _random_search_weights(objective, covariance.index, max_weight, attempts=1200)
    return _weights_from_array(best_weights, covariance.index, max_weight)


def _weights_from_array(values: np.ndarray | pd.Series, index: pd.Index, max_weight: float) -> pd.Series:
    """Project raw optimizer output to valid long-only capped weights."""
    raw = np.asarray(values, dtype=float)
    if raw.shape[0] != len(index) or not np.all(np.isfinite(raw)):
        raise PortfolioError("Optimizer returned invalid weights.")
    weights = _project_to_capped_simplex(raw, max_weight)
    if not np.all(np.isfinite(weights)) or np.any(weights < -1e-8):
        raise PortfolioError("Optimizer returned invalid weights.")
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-8):
        raise PortfolioError("Optimizer returned weights that do not sum to 1.")
    return pd.Series(weights, index=index, dtype=float)


def _project_to_capped_simplex(values: np.ndarray, max_weight: float) -> np.ndarray:
    """Normalize weights onto the fully invested simplex with an upper cap."""
    values = np.asarray(values, dtype=float)
    n_assets = values.shape[0]
    validate_weight_bounds(n_assets, max_weight)
    if not np.all(np.isfinite(values)):
        values = np.full(n_assets, 1.0 / n_assets)

    lower = float(np.min(values) - max_weight)
    upper = float(np.max(values))
    for _ in range(100):
        midpoint = (lower + upper) / 2
        projected = np.clip(values - midpoint, 0.0, max_weight)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint

    projected = np.clip(values - upper, 0.0, max_weight)
    total = float(projected.sum())
    if np.isclose(total, 0.0):
        projected = np.full(n_assets, 1.0 / n_assets)
    else:
        projected = projected / total

    for _ in range(10):
        excess = np.maximum(projected - max_weight, 0.0)
        if excess.sum() <= 1e-12:
            break
        projected = np.minimum(projected, max_weight)
        room = max_weight - projected
        room[room < 0] = 0
        if room.sum() <= 1e-12:
            break
        projected += room / room.sum() * excess.sum()
    return projected / projected.sum()
