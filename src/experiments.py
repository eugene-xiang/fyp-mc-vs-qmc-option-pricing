"""Edit SETTINGS below, then run: python -m src.experiments"""
from pathlib import Path
from copy import deepcopy
import csv
import json

from .models import AsianContract, positive_integer, european_call_price, deterministic_price
from .pricing import price_option, summarise, tune_shift, ENDPOINT_EPSILON, SOBOL_BITS, METHODS

ROOT = Path(__file__).resolve().parents[1]

# Single-contract settings
SETTINGS = {
    "contract": {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "monitoring_dates": 64,
    },
    "methods": ["mc", "rqmc", "mc_av", "rqmc_av", "mc_is", "rqmc_is"],
    "budgets": [1024, 4096, 16384],                                 # Total payoff evaluations, not base points.
    "replications": 32,                                             # Independent price estimates per method and budget.
    "seed": 42044026,
    "importance_sampling": {
        "shift_candidates": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "pilot_points": 4096,
    },
    "benchmark_points": 262144,
    "benchmark_replications": 16,
    "benchmark_se_fraction": 0.1,                                   # Reference SE should be at most 10% of RMSE.
}

SWEEP_SETTINGS = {
    "moneyness_strikes": [90.0, 100.0, 110.0, 120.0],
    "moneyness_dimension": 64,
    "dimension_strikes": [90.0, 100.0, 120.0],
    "dimensions": [16, 64, 128],
    "seed": 42045026,
    
    # Extra independent reference replicates for the most accurate OTM IS rows.
    "benchmark_extensions": {
        "strike-120-dates-128": {"replications": 48, "seed": 44045026},
    },
}

# Small analytical checks for the report. The European settings match the test.
VALIDATION_SETTINGS = {
    "european": {
        "contract": {**SETTINGS["contract"], "monitoring_dates": 1},
        "budget": 16384, "replications": 16, "seed": 729, "is_shift": 1.0,
    },
    "deterministic": {
        "contract": {**SETTINGS["contract"], "volatility": 0.0, "monitoring_dates": 16},
        "strikes": [80.0, 100.0, 120.0], "budget": 256, "seed": 77,
    },
}


def new_run_folder(results_root, name):
    """Create a fresh folder, retaining every previous run."""
    folder = Path(results_root) / name
    number = 2
    while folder.exists():
        folder = Path(results_root) / f"{name}-{number}"
        number += 1
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def run_validation(settings=None, results_root=ROOT / "results"):
    """Save small analytical checks, separately from the research sweeps.
    The European case replays the mathematical test's common seeds across
    methods. These rows demonstrate correctness, not independent rankings.
    """
    settings = deepcopy(VALIDATION_SETTINGS if settings is None else settings)
    folder = new_run_folder(results_root, "validation")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    european = settings["european"]
    deterministic = settings["deterministic"]
    cases = [("european", european["contract"], european["budget"],
              european["replications"], european["seed"], european["is_shift"])]
    cases += [("deterministic", {**deterministic["contract"], "strike": strike}, deterministic["budget"], 1, deterministic["seed"], 0.0) for strike in deterministic["strikes"]]
    
    rows, summaries = [], []
    
    for case, parameters, budget, replications, seed, is_shift in cases:
        contract = AsianContract(**parameters)
        reference = european_call_price(contract) if case == "european" else deterministic_price(contract)
        
        for method in METHODS:
            n = budget // 2 if method.endswith("_av") else budget
            shift = is_shift if method.endswith("_is") else 0.0
            
            estimates = []
            for replicate in range(replications):
                result = price_option(contract, method, n, seed + replicate, shift=shift)
                estimates.append(result["estimate"])
                rows.append({"case": case, "strike": contract.strike, "method": method, "reference": reference, "replicate": replicate, "seed": seed + replicate, 
                             "base_points": n, "antithetic_pairs": n if method.endswith("_av") else 0, "payoff_evaluations": budget, "shift": shift, **result})
            
            if case == "european":
                summary = summarise(estimates, reference)
                mean, se = summary["mean"], summary["mean_standard_error"]
                
                if abs(mean - reference) > 6 * se:
                    raise ValueError("European validation exceeds the test's six-SE tolerance")
            else:
                mean, se = estimates[0], 0.0                        # Analytically deterministic at shift zero.
                
                if abs(mean - reference) > 1e-12:
                    raise ValueError("Deterministic validation exceeds the numerical tolerance")
            summaries.append({"case": case, "strike": contract.strike, "method": method, "reference": reference, "mean": mean, "mean_standard_error": se,
                              "replications": replications, "absolute_error_of_mean": abs(mean-reference)})
    
    for name, data in (("replicates.csv", rows), ("summary.csv", summaries)):
        with (folder / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    
    return folder


def sweep_contracts(settings, sweep):
    """List unique contracts, with the moneyness sweep first."""
    positive_integer(sweep["seed"], "sweep seed")
    
    for key in ("moneyness_strikes", "dimension_strikes", "dimensions"):
        values = sweep[key]
        if not values or values != sorted(set(values)):
            raise ValueError(f"{key} must be increasing, unique and nonempty")
    
    pairs = [(strike, sweep["moneyness_dimension"]) for strike in sweep["moneyness_strikes"]]
    pairs += [(strike, dates) for dates in sweep["dimensions"] for strike in sweep["dimension_strikes"]]
    
    contracts = []
    for strike, dates in dict.fromkeys(pairs):
        contract = {**settings["contract"], "strike": strike, "monitoring_dates": dates}
        AsianContract(**contract)                               # Validate the whole sweep before simulation.
        contracts.append(contract)
    
    return contracts


def contract_name(contract):
    return f"strike-{contract['strike']:g}-dates-{contract['monitoring_dates']}"


def summarise_sweep(folder):
    """Combine contract summaries from saved replicates."""
    folder = Path(folder)
    saved = json.loads((folder / "settings.json").read_text())
    
    rows = []
    for contract in sweep_contracts(saved["experiment"], saved["sweep"]):
        name = contract_name(contract)
        summary = summarise_results(folder / name)
        for result in summary["comparisons"]:
            rows.append({"contract": name, "strike": contract["strike"], "monitoring_dates": contract["monitoring_dates"], **result})
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    
    return rows


def run_sweep(settings=None, sweep=None, results_root=ROOT / "results"):
    """Run each unique contract once, reserving disjoint blocks of seeds."""
    settings = deepcopy(SETTINGS if settings is None else settings)
    sweep = deepcopy(SWEEP_SETTINGS if sweep is None else sweep)
    methods = settings.get("methods", ["mc", "rqmc"])
    validate_settings(settings, methods)
    contracts = sweep_contracts(settings, sweep)
    extensions = sweep.get("benchmark_extensions", {})
    
    if set(extensions) - {contract_name(c) for c in contracts}:
        raise ValueError("Benchmark extension names must belong to the sweep")
    
    seeds_per_contract = (int(any(m.endswith("_is") for m in methods)) + settings["benchmark_replications"]
                          + len(settings["budgets"]) * len(methods) * settings["replications"])
    reserved = set(range(sweep["seed"], sweep["seed"] + len(contracts) * seeds_per_contract))
    
    for extension in extensions.values():
        positive_integer(extension["replications"], "additional benchmark replications")
        positive_integer(extension["seed"], "additional benchmark seed")
        extra_seeds = set(range(extension["seed"], extension["seed"] + extension["replications"]))
        if reserved & extra_seeds:
            raise ValueError("Additional benchmark seeds overlap another stream")
        reserved.update(extra_seeds)
    folder = new_run_folder(results_root, "sweep")
    (folder / "settings.json").write_text(json.dumps({"experiment": settings, "sweep": sweep}, indent=2, allow_nan=False) + "\n")
    
    # Exactly one seed per pilot, benchmark replicate and evaluation replicate.
    for index, contract in enumerate(contracts):
        current = deepcopy(settings)
        current["contract"] = contract
        current["seed"] = sweep["seed"] + index * seeds_per_contract
        if contract_name(contract) in extensions:
            current["benchmark_extension"] = extensions[contract_name(contract)]
        print(f"Contract {index + 1}/{len(contracts)}: {contract_name(contract)}", flush=True)
        run_experiment(current, folder, run_name=contract_name(contract))
    
    return folder, summarise_sweep(folder)


def summarise_results(folder):
    """Recalculate tables from the saved CSV."""
    folder = Path(folder)
    saved_settings = json.loads((folder / "settings.json").read_text())
    settings = saved_settings["experiment"]
    pilot = saved_settings.get("pilot")
    with (folder / "replicates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference_rows = [r for r in rows if r["phase"] == "benchmark"]
    expected_reference_count = (settings["benchmark_replications"]
                                + settings.get("benchmark_extension", {}).get("replications", 0))
    if len(reference_rows) != expected_reference_count:
        raise ValueError("Benchmark data is incomplete")
    reference = summarise([float(r["estimate"]) for r in reference_rows])
    reference["base_points"] = settings["benchmark_points"]
    
    # The original baseline run was saved before settings included a method list.
    methods = settings.get("methods", ["mc", "rqmc"])
    comparisons = []
    for budget in settings["budgets"]:
        for method in methods:
            group = [r for r in rows if r["phase"] == "evaluation"
                     and r["method"] == method and int(r["payoff_evaluations"]) == budget]
            if len(group) != settings["replications"]:
                raise ValueError("Evaluation data is incomplete")
            result = summarise([float(r["estimate"]) for r in group], reference["mean"])
            runtimes = [float(row["elapsed_seconds"]) for row in group]
            allowed_benchmark_error = settings["benchmark_se_fraction"] * result["replicate_rmse"]
            result["method"] = method
            result["payoff_budget"] = budget
            result["mean_elapsed_seconds"] = sum(runtimes) / len(runtimes)
            if pilot is not None:
                is_method = method.endswith("_is")
                result["selected_shift"] = pilot["selected_shift"] if is_method else 0.0
                result["pilot_payoff_evaluations"] = pilot["payoff_evaluations"] if is_method else 0
                result["pilot_elapsed_seconds"] = pilot["elapsed_seconds"] if is_method else 0.0
                # Standalone cost: one pilot plus one evaluation replicate.
                # The actual experiment shares one pilot across IS methods and budgets.
                result["standalone_payoff_evaluations"] = budget + result["pilot_payoff_evaluations"]
                result["standalone_elapsed_seconds"] = ( result["mean_elapsed_seconds"] + result["pilot_elapsed_seconds"] )
            result["benchmark_precision_sufficient"] = ( reference["mean_standard_error"] <= allowed_benchmark_error )
            # Include reference statistics in the CSV so it is self-contained.
            for name, value in reference.items():
                result[f"benchmark_{name}"] = value
            comparisons.append(result)
    for result in comparisons:
        baseline_method = result["method"].split("_")[0]
        baseline = next(row for row in comparisons
                        if row["method"] == baseline_method
                        and row["payoff_budget"] == result["payoff_budget"])
        variance = result["estimator_variance"]
        if variance > 0:
            result["variance_reduction_factor"] = baseline["estimator_variance"] / variance
        else:
            result["variance_reduction_factor"] = None
        result["variance_reduction_factor_defined"] = variance > 0
    summary = {"benchmark": reference, "comparisons": comparisons}
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    
    return summary


def save_replicate(writer, phase, method, replicate, seed, n, result, shift=0.0):
    """Write one simulation result. This function does not calculate prices."""
    row = {
        "phase": phase,
        "method": method,
        "replicate": replicate,
        "seed": seed,
        "base_points": n,
        "antithetic_pairs": n if method.endswith("_av") else 0,
        "payoff_evaluations": 2 * n if method.endswith("_av") else n,
        "shift": shift,
        **result,
    }
    writer.writerow(row)


def write_benchmark_extension(writer, settings):
    """Append independent reference estimates without changing evaluation inputs."""
    extension = settings["benchmark_extension"]
    contract = AsianContract(**settings["contract"])
    n = settings["benchmark_points"]
    print(f"Additional benchmark: {extension['replications']} replicates of {n} paths", flush=True)
    for index in range(extension["replications"]):
        seed = extension["seed"] + index
        result = price_option(contract, "rqmc", n, seed)
        replicate = settings["benchmark_replications"] + index
        save_replicate(writer, "benchmark", "rqmc", replicate, seed, n, result)


def extend_benchmark(folder, replications, seed):
    """Strengthen one saved reference, keeping every original replicate row.
    One extension per contract is supported. Its explicit settings reproduce
    the added reference while retaining the original pilot/evaluation seeds.
    """
    folder = Path(folder)
    saved = json.loads((folder / "settings.json").read_text())
    settings = saved["experiment"]
    if "benchmark_extension" in settings:
        raise ValueError("This contract already has a benchmark extension")
    extension = {"replications": replications, "seed": seed}
    settings["benchmark_extension"] = extension
    validate_settings(settings, settings.get("methods", ["mc", "rqmc"]))
    
    # Include sibling contracts when this is a sweep.
    parent_settings_path = folder.parent / "settings.json"
    parent_settings = json.loads(parent_settings_path.read_text()) if parent_settings_path.exists() else {}
    folders = [folder]
    if "sweep" in parent_settings:
        folders = [p.parent for p in folder.parent.glob("*/replicates.csv")]
    used_seeds = set()
    for child in folders:
        child_settings = json.loads((child / "settings.json").read_text())
        if "pilot" in child_settings:
            used_seeds.add(child_settings["pilot"]["seed"])
        with (child / "replicates.csv").open(newline="") as handle:
            used_seeds.update(int(row["seed"]) for row in csv.DictReader(handle))
    if used_seeds.intersection(range(seed, seed + replications)):
        raise ValueError("Additional benchmark seeds overlap saved streams")
    with (folder / "replicates.csv").open(newline="") as handle:
        fields = next(csv.reader(handle))
    with (folder / "replicates.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        write_benchmark_extension(writer, settings)
    (folder / "settings.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    if "sweep" in parent_settings:
        parent_settings["sweep"].setdefault("benchmark_extensions", {})[folder.name] = extension
        parent_settings_path.write_text(json.dumps(parent_settings, indent=2, allow_nan=False) + "\n")
    
    return summarise_results(folder)


def validate_settings(settings, methods):
    """Check the sampling configuration before starting a run."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be a nonempty, unique selection from {METHODS}")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"methods must be a nonempty, unique selection from {METHODS}")
        baseline_method = method.split("_")[0]
        if baseline_method not in methods:
            raise ValueError("Each modified method requires its corresponding baseline")
    for count in [settings["replications"], settings["benchmark_replications"]]:
        positive_integer(count, "replications")
        if count < 2:
            raise ValueError("At least two replicates are required")
    positive_integer(settings["seed"], "seed")
    budgets = settings["budgets"]
    if not budgets or budgets != sorted(set(budgets)):
        raise ValueError("Budgets must be increasing and unique")
    for n in budgets + [settings["benchmark_points"]]:
        positive_integer(n, "budget")
        if n & (n - 1) or n > 2**SOBOL_BITS:
            raise ValueError("Budgets must be complete power-of-two Sobol' nets")
    has_antithetics = any(method.endswith("_av") for method in methods)
    if has_antithetics and min(budgets) < 2:
        raise ValueError("Antithetic budgets must allow at least one pair")
    if not 0 < settings["benchmark_se_fraction"] < 1:
        raise ValueError("Benchmark SE fraction must be between zero and one")
    if "benchmark_extension" in settings:
        extension = settings["benchmark_extension"]
        positive_integer(extension["replications"], "additional benchmark replications")
        positive_integer(extension["seed"], "additional benchmark seed")
        initial_count = (int(any(m.endswith("_is") for m in methods)) + settings["benchmark_replications"] + len(budgets) * len(methods) * settings["replications"])
        if (extension["seed"] < settings["seed"] + initial_count
                and extension["seed"] + extension["replications"] > settings["seed"]):
            raise ValueError("Additional benchmark seeds overlap the initial experiment")


def run_experiment(settings=None, results_root=ROOT / "results", *, run_name=None):
    """Run one contract at several budgets, saving each independent replicate."""
    if settings is None:
        settings = SETTINGS
    contract = AsianContract(**settings["contract"])
    methods = settings.get("methods", ["mc", "rqmc"])
    validate_settings(settings, methods)
    budgets = settings["budgets"]
    has_antithetics = any(method.endswith("_av") for method in methods)
    has_importance_sampling = any(method.endswith("_is") for method in methods)

    # Tune once, before evaluating either IS method. No pilot inputs are reused by the reference or evaluation streams. Each will get a new seed.
    seed = settings["seed"]
    pilot = None
    if has_importance_sampling:
        tuning = settings["importance_sampling"]
        pilot = tune_shift(contract, tuning["shift_candidates"], tuning["pilot_points"], seed)
        seed += 1
        print(f"Pilot selected shift {pilot['selected_shift']} (informative: {pilot['informative']})", flush=True)

    # Keep every run without overwriting previous results.
    if run_name is not None:
        if Path(run_name).name != run_name or run_name in ("", ".", ".."):
            raise ValueError("run_name must be a folder name")
    elif has_antithetics and has_importance_sampling:
        run_name = "comparison"
    elif has_importance_sampling:
        run_name = "importance-sampling"
    elif has_antithetics:
        run_name = "antithetic"
    else:
        run_name = "baseline"
    folder = new_run_folder(results_root, run_name)
    run_settings = {
        "experiment": settings,
        "endpoint_epsilon": ENDPOINT_EPSILON,
        "sobol_bits": SOBOL_BITS,
        "seed_rule": "Start at seed; reserve the first seed for the IS pilot if used; then increment after each replicate",
        "antithetic_rule": "Negate the complete normal vector; average the two discounted payoffs",
        "endpoint_adjustments_rule": "Count clipped base uniform coordinates; reflected normals need no second transform",
    }
    
    if pilot is not None:
        run_settings["pilot"] = pilot
        run_settings["importance_sampling_rule"] = "theta = selected_shift / sqrt(d) * ones(d); ordinary likelihood-ratio weights"
        run_settings["pilot_reuse_rule"] = "One MC pilot shared by all IS evaluation budgets and replicates of both methods"
    (folder / "settings.json").write_text( json.dumps(run_settings, indent=2, allow_nan=False) + "\n" )
    fields = ["phase", "method", "replicate", "seed", "base_points", "antithetic_pairs",
              "payoff_evaluations", "shift", "estimate", "elapsed_seconds", "iid_standard_error", "endpoint_adjustments"]
    with (folder / "replicates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()

        # 1. Estimate a reference price with larger, independent Sobol' runs.
        n = settings["benchmark_points"]
        print(f"Benchmark: {n} paths per estimate", flush=True)
        for replicate in range(settings["benchmark_replications"]):
            result = price_option(contract, "rqmc", n, seed)
            save_replicate(writer, "benchmark", "rqmc", replicate, seed, n, result)
            seed += 1

        # 2. Match payoff evaluations, counting each antithetic pair twice.
        # The legacy method list and loop order still reproduce baseline runs.
        for budget in budgets:
            print(f"Evaluation: {budget} payoffs per estimate", flush=True)
            for replicate in range(settings["replications"]):
                for method in methods:
                    n = budget // 2 if method.endswith("_av") else budget
                    shift = pilot["selected_shift"] if method.endswith("_is") else 0.0
                    result = price_option(contract, method, n, seed, shift=shift)
                    save_replicate(writer, "evaluation", method, replicate, seed, n, result, shift)
                    seed += 1

        if "benchmark_extension" in settings:
            write_benchmark_extension(writer, settings)

    # 3. Calculate comparison metrics from the saved estimates.
    summary = summarise_results(folder)
    print(f"Saved to {folder}")
    print(f"Reference price: {summary['benchmark']['mean']:.6f}")
    
    return folder, summary


if __name__ == "__main__":
    from .plotting import plot_sweep, export_sweep_tables
    run_folder, _ = run_sweep()
    plot_sweep(run_folder)
    export_sweep_tables(run_folder)
