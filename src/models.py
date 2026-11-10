"""Shared exact Black-Scholes paths and discounted arithmetic Asian payoffs."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class AsianContract:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    volatility: float = 0.20
    monitoring_dates: int = 64

    def __post_init__(self):
        for name in ("spot", "strike", "maturity", "rate", "volatility"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if min(self.spot, self.strike, self.maturity) <= 0 or self.volatility < 0:
            raise ValueError("Spot, strike and maturity must be positive; volatility nonnegative")
        
        positive_integer(self.monitoring_dates, "monitoring_dates")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def asset_paths(normals, contract):
    """Return an array with one row per path and one column per monitoring date."""
    z = np.asarray(normals, dtype=np.float64)
    
    if z.ndim != 2 or z.shape[1] != contract.monitoring_dates or z.shape[0] == 0:
        raise ValueError("Normals must have shape (positive path count, monitoring_dates)")
    if not np.isfinite(z).all():
        raise ValueError("Normal inputs must be finite")
    
    dt = contract.maturity / contract.monitoring_dates
    
    with np.errstate(over="raise", invalid="raise", under="raise"):
        drift = (contract.rate - 0.5 * contract.volatility**2) * dt
        random_movements = contract.volatility * math.sqrt(dt) * z
        log_increments = drift + random_movements
        cumulative_log_returns = np.cumsum(log_increments, axis=1)
        paths = contract.spot * np.exp(cumulative_log_returns)
    
    if not np.isfinite(paths).all():
        raise FloatingPointError("Non-finite asset paths")
    
    return paths


def discounted_payoffs(paths, contract):
    paths = np.asarray(paths, dtype=np.float64)
    
    if paths.ndim != 2 or paths.shape[1] != contract.monitoring_dates or paths.shape[0] == 0:
        raise ValueError("Paths must have shape (positive path count, monitoring_dates)")
    if not np.isfinite(paths).all() or np.any(paths <= 0):
        raise ValueError("Asset prices must be finite and positive")
    
    with np.errstate(over="raise", invalid="raise", under="raise"):
        average_prices = paths.mean(axis=1)
        call_payoffs = np.maximum(average_prices - contract.strike, 0.0)
        discount_factor = np.exp(-contract.rate * contract.maturity)
        result = discount_factor * call_payoffs
    
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite discounted payoffs")
    
    return result


def payoffs_from_normals(normals, contract):
    paths = asset_paths(normals, contract)
    return discounted_payoffs(paths, contract)


def deterministic_price(contract):
    """Independent zero-volatility check using prices at absolute times."""
    if contract.volatility != 0:
        raise ValueError("The deterministic formula requires zero volatility")
    
    prices = []
    for date in range(1, contract.monitoring_dates + 1):
        growth = contract.rate * date * contract.maturity / contract.monitoring_dates
        prices.append(contract.spot * math.exp(growth))
    average_price = math.fsum(prices) / contract.monitoring_dates
    payoff = max(average_price - contract.strike, 0.0)

    return math.exp(-contract.rate * contract.maturity) * payoff


def european_call_price(contract):
    """Analytical reference for the one-monitoring-date contract only."""
    if contract.monitoring_dates != 1:
        raise ValueError("European reference requires one monitoring date")
    if contract.volatility == 0:
        return deterministic_price(contract)
    sigma_t = contract.volatility * math.sqrt(contract.maturity)
    log_moneyness = math.log(contract.spot / contract.strike)
    drift = (contract.rate + contract.volatility**2 / 2) * contract.maturity
    d1 = (log_moneyness + drift) / sigma_t
    d2 = d1 - sigma_t
    discounted_strike = contract.strike * math.exp(-contract.rate * contract.maturity)
    
    return float(contract.spot * ndtr(d1) - discounted_strike * ndtr(d2))
