"""Figures and report tables from saved results, without new simulation."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")                                   # Save figures without opening a window.
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from scipy.stats import t


METHOD_LABELS = {
    "mc": "MC",
    "rqmc": "RQMC",
    "mc_av": "MC + AV",
    "rqmc_av": "RQMC + AV",
    "mc_is": "MC + IS",
    "rqmc_is": "RQMC + IS",
}

METHOD_COLORS = dict(zip(METHOD_LABELS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def export_validation_tables(folder, output_folder):
    """Export analytical validation evidence from the saved small check run."""
    rows = read_summary(folder)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    european = []
    for row in rows:
        if row["case"] == "european":
            european.append(f"{METHOD_LABELS[row['method']]} & {float(row['mean']):.8f} & "
                            f"{float(row['mean_standard_error']):.8f} & "
                            f"{float(row['absolute_error_of_mean']):.8f}" + r" \\")
    deterministic = []
    for strike in sorted({float(r["strike"]) for r in rows if r["case"] == "deterministic"}):
        group = [r for r in rows if r["case"] == "deterministic" and float(r["strike"]) == strike]
        error = max(float(r["absolute_error_of_mean"]) for r in group)
        if error == 0:
            error_text = "0"
        else:
            mantissa, exponent = f"{error:.2e}".split("e")
            error_text = f"${mantissa}\\times10^{{{int(exponent)}}}$"
        deterministic.append(f"{strike:g} & {float(group[0]['reference']):.8f} & "
                             f"{float(group[0]['mean']):.8f} & {error_text}" + r" \\")
    for name, lines in (("validation_european.tex", european), ("validation_deterministic.tex", deterministic)):
        (output / name).write_text("% Generated from saved validation summary.csv.\n" + "\n".join(lines) + "\n")


def export_pilot_table(folder, destination):
    """Show zero-shift and selected-shift diagnostics for the moneyness sweep."""
    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())["sweep"]
    dates = settings["moneyness_dimension"]
    lines = []
    for strike in settings["moneyness_strikes"]:
        saved = json.loads((folder / f"strike-{strike:g}-dates-{dates}" / "settings.json").read_text())
        pilot = saved["pilot"]
        candidates = {row["shift"]: row for row in pilot["candidates"]}
        zero, selected = candidates[0.0], candidates[pilot["selected_shift"]]
        lines.append(f"{strike:g} & {pilot['selected_shift']:g} & {zero['positive_payoffs']} & "
                     f"{selected['positive_payoffs']} & {zero['second_moment']:.3f} & "
                     f"{selected['second_moment']:.3f}" + r" \\")
    Path(destination).write_text("% Generated from saved moneyness pilot diagnostics.\n" + "\n".join(lines) + "\n")


def plot_sweep(folder, output_folder=None, budget=None):
    """Compare moneyness and dimensions using only saved summary rows."""
    folder = Path(folder)
    output = Path(output_folder) if output_folder is not None else folder
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((folder / "settings.json").read_text())
    sweep = saved["sweep"]
    rows = read_summary(folder)
    budget = max(int(r["payoff_budget"]) for r in rows) if budget is None else budget
    selected = [r for r in rows if int(r["payoff_budget"]) == budget]
    if not selected:
        raise ValueError(f"No saved results at payoff budget {budget}")
    paths = []

    def curves(ax, group, xfield, yfield, methods, *, logx=False):
        for method in methods:
            points = sorted((r for r in group if r["method"] == method),
                            key=lambda r: float(r[xfield]))
            if not points:
                continue
            # Undefined variance ratios remain gaps, not artificial zeroes.
            y = [float(r[yfield]) if r[yfield] else float("nan") for r in points]
            ax.plot([float(r[xfield]) for r in points], y, "o-",
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method], markersize=4)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=.2)

    def save(fig, axes, filename, title):
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(title)
        path = output / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    strikes = sweep["moneyness_strikes"]
    dates = sweep["moneyness_dimension"]
    for time_plot in (False, True):
        nrows = (len(strikes) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(7, 2.8 * nrows + .6),
                                 squeeze=False, constrained_layout=True)
        for ax, strike in zip(axes.flat, strikes):
            group = [r for r in rows if float(r["strike"]) == strike
                     and int(r["monitoring_dates"]) == dates]
            xfield = "standalone_elapsed_seconds" if time_plot else "payoff_budget"
            # Baseline-only saved runs have no pilot cost field.
            if time_plot and xfield not in group[0]:
                xfield = "mean_elapsed_seconds"
            curves(ax, group, xfield, "replicate_rmse", METHOD_LABELS, logx=True)
            if time_plot:
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axhline(float(group[0]["benchmark_mean_standard_error"]),
                       color="gray", linestyle=":")
            ax.set_title(f"K = {strike:g}, d = {dates}")
            ax.set_ylabel("Replicate RMSE")
            ax.set_xlabel("Evaluation time + one pilot (s)" if time_plot else "Evaluation payoffs")
        for ax in list(axes.flat)[len(strikes):]:
            ax.set_visible(False)
        save(fig, list(axes.flat), "moneyness_runtime.png" if time_plot else "moneyness_convergence.png",
             "Moneyness comparison; dotted lines show benchmark SE")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.7), constrained_layout=True)
    group = [r for r in selected if int(r["monitoring_dates"]) == dates
             and float(r["strike"]) in strikes]
    for ax, baseline in zip(axes, ("mc", "rqmc")):
        curves(ax, group, "strike", "variance_reduction_factor", [baseline + "_av", baseline + "_is"])
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_title(METHOD_LABELS[baseline])
        ax.set_xlabel("Strike")
        ax.set_ylabel("Baseline variance / modified variance")
        ax.set_xticks(strikes)
        ax.legend(fontsize=8)
    fig.suptitle(f"Variance reduction at d = {dates}, B = {budget:,}; above 1 is better")
    path = output / "moneyness_variance.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    strikes = sweep["dimension_strikes"]
    fig, axes = plt.subplots(len(strikes), 2, figsize=(7, 2.5 * len(strikes) + .6), squeeze=False, constrained_layout=True)
    for pair, strike in zip(axes, strikes):
        group = [r for r in selected if float(r["strike"]) == strike and int(r["monitoring_dates"]) in sweep["dimensions"]]
        curves(pair[0], group, "monitoring_dates", "replicate_rmse", METHOD_LABELS)
        curves(pair[1], group, "monitoring_dates", "variance_reduction_factor",
               [m for m in METHOD_LABELS if "_" in m])
        pair[1].axhline(1, color="gray", linestyle="--")
        for ax in pair:
            ax.set_title(f"K = {strike:g}")
            ax.set_xlabel("Monitoring dates")
            ax.set_xticks(sweep["dimensions"])
        pair[0].set_ylabel("Replicate RMSE")
        pair[1].set_ylabel("Variance-reduction factor")
    save(fig, list(axes.flat), "dimension_comparison.png",
         f"Monitoring-frequency comparison at B = {budget:,}")
    return paths


def export_sweep_tables(folder, output_folder=None, budget=None):
    """Write rows for the main report from saved summaries."""
    folder = Path(folder)
    output = Path(output_folder) if output_folder is not None else folder
    output.mkdir(parents=True, exist_ok=True)
    rows = read_summary(folder)
    budget = max(int(r["payoff_budget"]) for r in rows) if budget is None else budget
    rows = [r for r in rows if int(r["payoff_budget"]) == budget]
    
    if not rows:
        raise ValueError(f"No saved results at payoff budget {budget}")
    groups = {}
    for row in rows:
        key = (int(row["monitoring_dates"]), float(row["strike"]))
        groups.setdefault(key, {})[row["method"]] = row
    tables = {"sweep_benchmarks.tex": [], "sweep_variance.tex": [], "sweep_rmse.tex": []}
    for (dates, strike), group in sorted(groups.items()):
        base = group["mc"]
        prefix = f"{strike:g} & {dates} & "
        shift = float(group["mc_is"]["selected_shift"])
        tables["sweep_benchmarks.tex"].append(prefix + f"{float(base['benchmark_mean']):.6f} & {float(base['benchmark_mean_standard_error']):.6f} & {shift:g}")
        ratios = [group[m]["variance_reduction_factor"] for m in METHOD_LABELS if "_" in m]
        tables["sweep_variance.tex"].append(prefix + " & ".join(f"{float(value):.2f}" if value else "--" for value in ratios))
        tables["sweep_rmse.tex"].append(prefix + " & ".join(f"{float(group[m]['replicate_rmse']):.5f}" for m in METHOD_LABELS))
    for filename, lines in tables.items():
        (output / filename).write_text("% Generated from saved sweep summary.csv; evaluation budget " + str(budget) + ".\n" + "\n".join(line + r" \\" for line in lines) + "\n")
    
    return [output / name for name in tables]


def read_summary(folder):
    with (Path(folder) / "summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(folder, *, destination=None, payoff_only=False, methods=None):
    """Plot saved RMSE summaries. payoff_only gives the report's baseline panel."""
    folder = Path(folder)
    comparisons = read_summary(folder)
    if methods is not None:
        comparisons = [r for r in comparisons if r["method"] in methods]
    benchmark_standard_error = float(comparisons[0]["benchmark_mean_standard_error"])
    has_antithetics = any(r["method"].endswith("_av") for r in comparisons)
    has_importance_sampling = any(r["method"].endswith("_is") for r in comparisons)
    # Stack the comparison panels so labels remain readable in the report.
    if payoff_only:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axes = [ax]
    elif has_antithetics or has_importance_sampling:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    if has_importance_sampling:
        time_field = "standalone_elapsed_seconds"
        time_label = "Mean evaluation runtime + one pilot (seconds)"
        payoff_label = "Evaluation payoffs per replicate"
    else:
        time_field = "mean_elapsed_seconds"
        time_label = "Mean runtime per replicate (seconds)"
        payoff_label = "Payoff evaluations per replicate"

    for method, label in METHOD_LABELS.items():
        rows = [r for r in comparisons if r["method"] == method]
        if not rows:
            continue
        errors = [float(r["replicate_rmse"]) for r in rows]
        budgets = [int(r["payoff_budget"]) for r in rows]
        axes[0].loglog(budgets, errors, "o-", label=label)
        if not payoff_only:
            runtimes = [float(r[time_field]) for r in rows]
            axes[1].loglog(runtimes, errors, "o-", label=label)

    for ax in axes:
        if benchmark_standard_error > 0:
            ax.axhline(benchmark_standard_error, color="gray", linestyle="--", label="Benchmark SE")
        ax.set_ylabel("Replicate RMSE")
        ax.grid(True, which="both", alpha=.25)
    axes[0].set_xlabel(payoff_label)
    if not payoff_only:
        axes[1].set_xlabel(time_label)
        axes[1].xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=8)
    if has_antithetics and has_importance_sampling:
        title = "Six-method comparison at matched evaluation budgets"
    elif has_importance_sampling:
        title = "Importance sampling at matched evaluation budgets"
    elif has_antithetics:
        title = "Antithetic comparison at matched payoff budgets"
    else:
        title = "Preliminary baseline comparison"
    fig.suptitle(title)
    destination = Path(destination) if destination is not None else folder / "convergence.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)

    return destination


def export_report_table(folder, destination, *, methods=None):
    """Write table rows for prices, uncertainty, RMSE, runtime and VRF."""
    comparisons = read_summary(folder)
    lines = []
    for row in comparisons:
        if methods is not None and row["method"] not in methods:
            continue
        budget = format(int(row["payoff_budget"]), ",").replace(",", r"\,")
        vrf = "--"
        if row["method"] not in ("mc", "rqmc"):
            if row["variance_reduction_factor_defined"] == "True":
                vrf = f"{float(row['variance_reduction_factor']):.2f}"
            else:
                vrf = "undefined"
        lines.append(
            f"        {METHOD_LABELS[row['method']]} & ${budget}$ & {float(row['mean']):.5f} & "
            f"{float(row['mean_standard_error']):.5f} & {float(row['replicate_rmse']):.5f} & "
            f"{float(row['mean_elapsed_seconds']):.4f} & {vrf} " + r"\\"
        )
    destination = Path(destination)
    destination.write_text("% Generated from saved summary.csv by src.plotting.export_report_table.\n" + "\n".join(lines) + "\n")
    return destination


def plot_appendix_figures(folder, output_folder, budget=16384):
    """Plot mean-price intervals and standalone IS costs from saved summaries."""
    rows = {}
    for row in read_summary(folder):
        if int(row["payoff_budget"]) == budget:
            rows[row["method"]] = row
    for method in METHOD_LABELS:
        if method not in rows:
            raise ValueError(f"Missing method {method} at payoff budget {budget} in {folder}")
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    price_path = output_folder / "appendix_price_intervals.png"
    cost_path = output_folder / "appendix_tuning_cost.png"

    reference = rows["mc"]
    reference_mean = float(reference["benchmark_mean"])
    reference_se = float(reference["benchmark_mean_standard_error"])
    reference_count = int(reference["benchmark_replications"])
    reference_width = t.ppf(.975, reference_count - 1) * reference_se

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True,
                             constrained_layout=True)
    panels = [("av", "(a) Antithetic comparison"),
              ("is", "(b) Importance-sampling comparison")]
    for ax, (suffix, title) in zip(axes, panels):
        ax.axvspan(reference_mean - reference_width, reference_mean + reference_width,
                   color="gray", alpha=.15, label="Reference: approximate 95% interval")
        ax.axvline(reference_mean, color="gray", linestyle=":", label="Reference mean")
        methods = ["mc", "rqmc", "mc_" + suffix, "rqmc_" + suffix]
        for y, method in enumerate(methods):
            row = rows[method]
            mean = float(row["mean"])
            standard_error = float(row["mean_standard_error"])
            replications = int(row["replications"])
            half_width = t.ppf(.975, replications - 1) * standard_error
            ax.errorbar(mean, y, xerr=half_width, fmt="o", capsize=4)
        ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_title(f"{title}\nReference = {reference_mean:.6f}, SE = {reference_se:.6f}")
        ax.grid(axis="x", alpha=.2)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].set_xlabel("Mean option price across independent replicates")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.savefig(price_path, dpi=300)
    plt.close(fig)

    methods = ["mc", "mc_is", "rqmc", "rqmc_is"]
    evaluation = [1000 * float(rows[m]["mean_elapsed_seconds"]) for m in methods]
    pilot = [1000 * float(rows[m]["pilot_elapsed_seconds"]) for m in methods]
    totals = [evaluation_time + pilot_time for evaluation_time, pilot_time in zip(evaluation, pilot)]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    evaluation_bars = ax.barh(range(len(methods)), evaluation, label="Evaluation")
    pilot_bars = ax.barh(range(len(methods)), pilot, left=evaluation, label="One pilot")
    ax.bar_label(evaluation_bars, fmt="%.1f", label_type="center")
    pilot_labels = [f"+ {time:.1f}" if time > 0 else "" for time in pilot]
    ax.bar_label(pilot_bars, labels=pilot_labels, label_type="center")
    ax.bar_label(pilot_bars, labels=[f"{total:.1f}" for total in totals], padding=4)
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.18)                                              # Leave room for total-time labels.
    ax.set_xlabel("Mean time per standalone valuation (ms)")
    ax.set_title(f"Importance sampling: {budget:,} evaluation payoffs per replicate")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.savefig(cost_path, dpi=300)
    plt.close(fig)

    return price_path, cost_path
