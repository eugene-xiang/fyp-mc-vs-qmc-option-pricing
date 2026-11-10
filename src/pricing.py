"""MC and scrambled Sobol' pricing, antithetics, and Gaussian importance sampling."""
from time import perf_counter
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from .models import payoffs_from_normals, positive_integer


ENDPOINT_EPSILON = 2.0**-53
SOBOL_BITS = 52
METHODS = ("mc", "rqmc", "mc_av", "rqmc_av", "mc_is", "rqmc_is")


def normal_transform(uniforms, epsilon=ENDPOINT_EPSILON):
    """Convert uniforms to normals, keeping every point and counting clipping, but can introduce a small numerical bias."""
    u = np.asarray(uniforms, dtype=float)

    # Validate inputs
    if not 0 < epsilon < 0.5 or 1 - epsilon == 1:
        raise ValueError("epsilon must define a representable interval inside (0,1)")
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError("Uniform inputs must be finite and in [0,1]")

    changes = int(np.count_nonzero((u < epsilon) | (u > 1 - epsilon)))
    safe_uniforms = np.clip(u, epsilon, 1 - epsilon)
    normals = ndtri(safe_uniforms)                                              # Inverse standard normal CDF.

    if not np.isfinite(normals).all():
        raise FloatingPointError("Non-finite normal inputs")

    return normals, changes


def sample_normals(n, dimension, method, seed, epsilon=ENDPOINT_EPSILON):
    """Return one normal-input row per path. The two methods differ only here."""
    positive_integer(n, "n")
    positive_integer(dimension, "dimension")

    if method == "mc":
        generator = np.random.Generator(np.random.PCG64(seed))                  # Initialise reproducible random number generator using PCG64
        uniforms = generator.random((n, dimension))
    elif method == "rqmc":
        if n & (n - 1) or n > 2**SOBOL_BITS:
            raise ValueError("Sobol' requires a power-of-two sample count within its bit depth")
        generator = qmc.Sobol(d=dimension, scramble=True, bits=SOBOL_BITS, rng=seed, optimization=None)
        uniforms = generator.random_base2(n.bit_length() - 1)
    else:
        raise ValueError("method must be 'mc' or 'rqmc'")
    
    return normal_transform(uniforms, epsilon)


def importance_payoffs(normals, contract, shift):
    """Return g(Z + theta) * exp(-theta.Z - |theta|^2/2).
    The scalar shift is the length of theta = shift / sqrt(d) * ones(d).
    This is an ordinary weighted mean, not a self-normalised estimator.
    """
    # Validate shift and normals
    if isinstance(shift, bool) or not isinstance(shift, (int, float)):
        raise ValueError("shift must be a finite nonnegative number")
    if not math.isfinite(shift) or shift < 0:
        raise ValueError("shift must be a finite nonnegative number")
    z = np.asarray(normals, dtype=float)
    if z.ndim != 2 or z.shape[1] != contract.monitoring_dates or z.shape[0] == 0:
        raise ValueError("Normals must have shape (positive path count, monitoring_dates)")
    if not np.isfinite(z).all():
        raise ValueError("Normal inputs must be finite")

    # Also makes zero-shift recovery exactly equal to the baseline calculation.
    if shift == 0:
        return payoffs_from_normals(z, contract)
    theta = np.full(contract.monitoring_dates, shift / math.sqrt(contract.monitoring_dates))
    with np.errstate(over="raise", under="raise", invalid="raise"):
        log_weights = -(z @ theta) - 0.5 * np.dot(theta, theta)
        weights = np.exp(log_weights)
        payoffs = payoffs_from_normals(z + theta, contract)
        weighted_payoffs = payoffs * weights
    
    if not np.isfinite(weighted_payoffs).all():
        raise FloatingPointError("Non-finite importance-sampling payoffs")
    
    return weighted_payoffs


def tune_shift(contract, candidates, n, seed):
    """Choose a scalar shift using one independent MC pilot shared by candidates.
    Minimise the estimated second moment among candidates with positive payoffs.
    Return diagnostics as well as the choice, and an all-zero pilot falls back to zero.
    """
    # Validate inputs
    if not candidates:
        raise ValueError("Shift candidates must be finite nonnegative numbers")
    for shift in candidates:
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise ValueError("Shift candidates must be finite nonnegative numbers")
        if not math.isfinite(shift) or shift < 0:
            raise ValueError("Shift candidates must be finite nonnegative numbers")
    if candidates != sorted(set(candidates)) or candidates[0] != 0:
        raise ValueError("Shift candidates must be increasing, unique, and include zero")
    
    start = perf_counter()
    normals, changes = sample_normals(n, contract.monitoring_dates, "mc", seed)
    diagnostics = []
    for shift in candidates:
        weighted = importance_payoffs(normals, contract, shift)
        with np.errstate(over="raise", under="raise", invalid="raise"):
            second_moment = float(np.mean(weighted**2))
        diagnostics.append({"shift": shift, "second_moment": second_moment, "positive_payoffs": int(np.count_nonzero(weighted > 0))})
    informative = [row for row in diagnostics if row["positive_payoffs"] > 0]
    
    # Candidates are sorted, so exact ties favour the smaller shift.
    if informative:
        selected = min(informative, key=lambda row: row["second_moment"])
        selected_shift = selected["shift"]
    else:
        selected_shift = 0.0
    
    return {
        "seed": seed,
        "base_points": n,
        "payoff_evaluations": n * len(candidates),
        "endpoint_adjustments": changes,
        "elapsed_seconds": perf_counter() - start,
        "selected_shift": selected_shift,
        "informative": bool(informative),
        "candidates": diagnostics,
    }


def price_option(contract, method, n, seed, epsilon=ENDPOINT_EPSILON, *, shift=0.0):

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not method.endswith("_is") and shift != 0:
        raise ValueError("A nonzero shift requires an importance-sampling method")
    
    start = perf_counter()
    sampler = method.split("_")[0]

    normals, changes = sample_normals(n, contract.monitoring_dates, sampler, seed, epsilon)

    if method.endswith("_is"):
        payoffs = importance_payoffs(normals, contract, shift)
    elif method.endswith("_av"):
        payoffs = payoffs_from_normals(normals, contract)
        opposite_payoffs = payoffs_from_normals(-normals, contract)
        payoffs = 0.5 * payoffs + 0.5 * opposite_payoffs
    else:
        payoffs = payoffs_from_normals(normals, contract)

    with np.errstate(over="raise", invalid="raise"):
        price = float(payoffs.mean())

    # MC observations are payoffs, pair averages, or weighted payoffs. Sobol' rows remain dependent, so no RQMC method gets an IID standard error.
    standard_error = None
    if sampler == "mc" and n > 1:
        with np.errstate(over="raise", invalid="raise"):
            standard_error = float(payoffs.std(ddof=1) / math.sqrt(n))
    
    return {
        "estimate": price,
        "elapsed_seconds": perf_counter() - start,
        "iid_standard_error": standard_error,
        "endpoint_adjustments": changes,
    }


def summarise(estimates, reference=None):
    """Summarise independent replicate prices, not individual Sobol' payoffs."""
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite replicate prices are required")

    # Each value is a complete simulation's price estimate.
    replication_count = len(values)
    mean_price = float(values.mean())
    variance = float(values.var(ddof=1))
    result = {
        "mean": mean_price,
        "estimator_variance": variance,
        "mean_standard_error": math.sqrt(variance / replication_count),
        "replications": replication_count,
    }

    if reference is not None:
        if not math.isfinite(reference):
            raise ValueError("Reference must be finite")
        errors = values - reference
        result["absolute_error_of_mean"] = abs(mean_price - reference)
        result["replicate_rmse"] = float(np.sqrt(np.mean(errors**2)))
    
    return result
