import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    BASE_DIR / "results" / "optimization"
)

PLOTS_DIR = (
    BASE_DIR / "results" / "plots"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD MULTI-RUN RESULTS
# ============================================================

INPUT_FILE = (
    RESULTS_DIR / "multi_run_results.csv"
)

df = pd.read_csv(INPUT_FILE)

print("\nLoaded multi-run results:")
print(df)


# ============================================================
# 3. STATISTICAL SUMMARY
# ============================================================

summary = (
    df.groupby("algorithm")
    .agg(
        runs=("seed", "count"),

        mean_energy_mwh=(
            "total_energy_mwh",
            "mean"
        ),

        std_energy_mwh=(
            "total_energy_mwh",
            "std"
        ),

        min_energy_mwh=(
            "total_energy_mwh",
            "min"
        ),

        max_energy_mwh=(
            "total_energy_mwh",
            "max"
        ),

        mean_runtime_seconds=(
            "execution_time_seconds",
            "mean"
        ),

        std_runtime_seconds=(
            "execution_time_seconds",
            "std"
        )
    )
    .reset_index()
)


# ============================================================
# 4. PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL STATISTICAL SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 5. MEAN ENERGY COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

algorithms = summary["algorithm"]

mean_energy = (
    summary["mean_energy_mwh"]
)

std_energy = (
    summary["std_energy_mwh"]
)

x = range(
    len(algorithms)
)

ax.bar(
    x,
    mean_energy,
    yerr=std_energy,
    capsize=6
)

ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    algorithms
)

ax.set_ylabel(
    "Mean Total Energy (MWh)"
)

ax.set_title(
    "PSO vs WOA: Mean Hydropower Generation"
)

fig.tight_layout()

energy_plot = (
    PLOTS_DIR
    / "pso_vs_woa_mean_energy.png"
)

fig.savefig(
    energy_plot,
    dpi=300
)

plt.close(fig)


# ============================================================
# 6. RUN-WISE ENERGY COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 5)
)

for algorithm in [
    "PSO",
    "WOA"
]:

    algorithm_df = (
        df[
            df["algorithm"] == algorithm
        ]
        .sort_values("seed")
    )

    ax.plot(
        range(
            1,
            len(algorithm_df) + 1
        ),
        algorithm_df[
            "total_energy_mwh"
        ],
        marker="o",
        label=algorithm
    )


ax.set_xlabel(
    "Run"
)

ax.set_ylabel(
    "Total Energy (MWh)"
)

ax.set_title(
    "Run-wise PSO vs WOA Energy Comparison"
)

ax.set_xticks(
    range(1, 6)
)

ax.legend()

fig.tight_layout()

runwise_plot = (
    PLOTS_DIR
    / "pso_vs_woa_runwise_energy.png"
)

fig.savefig(
    runwise_plot,
    dpi=300
)

plt.close(fig)


# ============================================================
# 7. RUNTIME COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

mean_runtime = (
    summary[
        "mean_runtime_seconds"
    ]
)

std_runtime = (
    summary[
        "std_runtime_seconds"
    ]
)

ax.bar(
    x,
    mean_runtime,
    yerr=std_runtime,
    capsize=6
)

ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    algorithms
)

ax.set_ylabel(
    "Mean Execution Time (seconds)"
)

ax.set_title(
    "PSO vs WOA: Mean Execution Time"
)

fig.tight_layout()

runtime_plot = (
    PLOTS_DIR
    / "pso_vs_woa_runtime.png"
)

fig.savefig(
    runtime_plot,
    dpi=300
)

plt.close(fig)


# ============================================================
# 8. ENERGY VARIABILITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for i, row in summary.iterrows():

    mean_value = (
        row["mean_energy_mwh"]
    )

    lower_error = (
        mean_value
        - row["min_energy_mwh"]
    )

    upper_error = (
        row["max_energy_mwh"]
        - mean_value
    )

    ax.errorbar(
        i,
        mean_value,
        yerr=[
            [lower_error],
            [upper_error]
        ],
        fmt="o",
        capsize=6
    )


ax.set_xticks(
    range(len(summary))
)

ax.set_xticklabels(
    summary["algorithm"]
)

ax.set_ylabel(
    "Total Energy (MWh)"
)

ax.set_title(
    "PSO vs WOA: Energy Variability Across Runs"
)

fig.tight_layout()

variability_plot = (
    PLOTS_DIR
    / "pso_vs_woa_energy_variability.png"
)

fig.savefig(
    variability_plot,
    dpi=300
)

plt.close(fig)


# ============================================================
# 9. REPORT-READY TABLE
# ============================================================

pso = summary[
    summary["algorithm"] == "PSO"
].iloc[0]

woa = summary[
    summary["algorithm"] == "WOA"
].iloc[0]


report_table = pd.DataFrame({

    "Metric": [

        "Mean Energy (MWh)",

        "Standard Deviation (MWh)",

        "Minimum Energy (MWh)",

        "Maximum Energy (MWh)",

        "Mean Runtime (seconds)",

        "Runtime Std. Dev. (seconds)",

        "Feasibility Rate (%)"

    ],

    "PSO": [

        pso["mean_energy_mwh"],

        pso["std_energy_mwh"],

        pso["min_energy_mwh"],

        pso["max_energy_mwh"],

        pso["mean_runtime_seconds"],

        pso["std_runtime_seconds"],

        100.0

    ],

    "WOA": [

        woa["mean_energy_mwh"],

        woa["std_energy_mwh"],

        woa["min_energy_mwh"],

        woa["max_energy_mwh"],

        woa["mean_runtime_seconds"],

        woa["std_runtime_seconds"],

        100.0

    ]

})


report_file = (
    RESULTS_DIR
    / "final_pso_woa_report_table.csv"
)

report_table.to_csv(
    report_file,
    index=False
)


# ============================================================
# 10. FINAL COMPARISON
# ============================================================

energy_difference = (
    woa["mean_energy_mwh"]
    - pso["mean_energy_mwh"]
)

energy_difference_percent = (
    energy_difference
    / pso["mean_energy_mwh"]
    * 100
)

runtime_difference = (
    woa["mean_runtime_seconds"]
    - pso["mean_runtime_seconds"]
)

runtime_difference_percent = (
    runtime_difference
    / pso["mean_runtime_seconds"]
    * 100
)


# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PSO vs WOA FINAL COMPARISON")
print("=" * 70)

print(
    f"\nPSO mean energy : "
    f"{pso['mean_energy_mwh']:.4f} MWh"
)

print(
    f"WOA mean energy : "
    f"{woa['mean_energy_mwh']:.4f} MWh"
)

print(
    f"Energy difference : "
    f"{energy_difference:.4f} MWh"
)

print(
    f"Energy difference : "
    f"{energy_difference_percent:.4f}%"
)


print(
    f"\nPSO mean runtime : "
    f"{pso['mean_runtime_seconds']:.4f} seconds"
)

print(
    f"WOA mean runtime : "
    f"{woa['mean_runtime_seconds']:.4f} seconds"
)

print(
    f"Runtime difference : "
    f"{runtime_difference:.4f} seconds"
)

print(
    f"Runtime difference : "
    f"{runtime_difference_percent:.4f}%"
)


print("\n")
print("=" * 70)
print("REPORT-READY TABLE")
print("=" * 70)

print(
    report_table.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    energy_plot
)

print(
    runwise_plot
)

print(
    runtime_plot
)

print(
    variability_plot
)

print(
    report_file
)

print("\nAnalysis completed successfully.")