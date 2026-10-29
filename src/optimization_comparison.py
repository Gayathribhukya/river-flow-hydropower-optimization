import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results/optimization"

PSO_SUMMARY = os.path.join(
    RESULTS_DIR,
    "pso_summary.csv"
)

WOA_SUMMARY = os.path.join(
    RESULTS_DIR,
    "woa_summary.csv"
)

DE_SUMMARY = os.path.join(
    RESULTS_DIR,
    "de_summary.csv"
)

PSO_CONVERGENCE = os.path.join(
    RESULTS_DIR,
    "pso_convergence.csv"
)

WOA_CONVERGENCE = os.path.join(
    RESULTS_DIR,
    "woa_convergence.csv"
)

DE_CONVERGENCE = os.path.join(
    RESULTS_DIR,
    "de_convergence.csv"
)

PSO_RESULT = os.path.join(
    RESULTS_DIR,
    "pso_hydropower_optimization.csv"
)

WOA_RESULT = os.path.join(
    RESULTS_DIR,
    "woa_hydropower_optimization.csv"
)

DE_RESULT = os.path.join(
    RESULTS_DIR,
    "de_hydropower_optimization.csv"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "de_pso_woa_comparison.csv"
)

MONTHLY_COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "de_pso_woa_monthly_comparison.csv"
)

PLOT_DIR = "results/plots"

CONVERGENCE_PLOT = os.path.join(
    PLOT_DIR,
    "de_pso_woa_convergence.png"
)

ENERGY_PLOT = os.path.join(
    PLOT_DIR,
    "de_pso_woa_energy_comparison.png"
)


# ============================================================
# LOAD SUMMARY
# ============================================================

def load_summary(path, algorithm):

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{algorithm} summary file is empty."
        )

    row = df.iloc[0].copy()

    row["algorithm"] = algorithm

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("DE vs PSO vs WOA OPTIMIZATION COMPARISON")
    print("=" * 70)

    # ========================================================
    # 1. LOAD RESULTS
    # ========================================================

    print("\nLoading optimization summaries...")

    de = load_summary(
        DE_SUMMARY,
        "DE"
    )

    pso = load_summary(
        PSO_SUMMARY,
        "PSO"
    )

    woa = load_summary(
        WOA_SUMMARY,
        "WOA"
    )

    # ========================================================
    # 2. CREATE COMPARISON TABLE
    # ========================================================

    comparison = pd.DataFrame([
    de,
    pso,
    woa
]).reset_index(drop=True)

    # Keep important columns
    columns = [
        "algorithm",
        "best_fitness",
        "total_energy_mwh",
        "total_spill_m3",
        "storage_violation_m3",
        "min_storage_m3",
        "max_storage_m3",
        "population_size",
        "iterations",
        "execution_time_seconds"
    ]

    available_columns = [
        col for col in columns
        if col in comparison.columns
    ]

    comparison = comparison[
        available_columns
    ]

    # ========================================================
    # 3. RANKING
    # ========================================================

    comparison["energy_rank"] = (
        comparison[
            "total_energy_mwh"
        ]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    comparison["fitness_rank"] = (
        comparison[
            "best_fitness"
        ]
        .rank(
            ascending=True,
            method="min"
        )
        .astype(int)
    )

    comparison["runtime_rank"] = (
        comparison[
            "execution_time_seconds"
        ]
        .rank(
            ascending=True,
            method="min"
        )
        .astype(int)
    )

    comparison["overall_rank_score"] = (
        comparison["energy_rank"]
        + comparison["fitness_rank"]
        + comparison["runtime_rank"]
    )

    comparison["overall_rank"] = (
        comparison[
            "overall_rank_score"
        ]
        .rank(
            ascending=True,
            method="min"
        )
        .astype(int)
    )

    # ========================================================
    # 4. FEASIBILITY CHECK
    # ========================================================

    # Small floating-point errors are treated as zero.
    TOLERANCE = 1e-6

    comparison["feasible"] = (
        comparison[
            "storage_violation_m3"
        ].abs()
        <= TOLERANCE
    )

    # ========================================================
    # 5. SAVE COMPARISON
    # ========================================================

    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # ========================================================
    # 6. PRINT COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("OPTIMIZATION PERFORMANCE")
    print("=" * 70)

    display_columns = [
        "algorithm",
        "total_energy_mwh",
        "best_fitness",
        "execution_time_seconds",
        "total_spill_m3",
        "storage_violation_m3",
        "feasible",
        "overall_rank"
    ]

    print(
        comparison[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # 7. BEST ALGORITHM
    # ========================================================

    best_energy_index = comparison[
        "total_energy_mwh"
    ].idxmax()

    best_energy_algorithm = comparison.loc[
        best_energy_index,
        "algorithm"
    ]

    best_energy = comparison.loc[
        best_energy_index,
        "total_energy_mwh"
    ]

    fastest_index = comparison[
        "execution_time_seconds"
    ].idxmin()

    fastest_algorithm = comparison.loc[
        fastest_index,
        "algorithm"
    ]

    best_overall_index = comparison[
        "overall_rank_score"
    ].idxmin()

    best_overall_algorithm = comparison.loc[
        best_overall_index,
        "algorithm"
    ]

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Best energy generation : "
        f"{best_energy_algorithm}"
    )

    print(
        f"Highest energy         : "
        f"{best_energy:.4f} MWh"
    )

    print(
        f"Fastest algorithm      : "
        f"{fastest_algorithm}"
    )

    print(
        f"Best overall ranking   : "
        f"{best_overall_algorithm}"
    )

    # ========================================================
    # 8. ENERGY DIFFERENCES
    # ========================================================

    de_energy = float(
        de["total_energy_mwh"]
    )

    pso_energy = float(
        pso["total_energy_mwh"]
    )

    woa_energy = float(
        woa["total_energy_mwh"]
    )

    print("\n" + "=" * 70)
    print("ENERGY COMPARISON")
    print("=" * 70)

    de_vs_pso = (
        de_energy - pso_energy
    )

    de_vs_woa = (
        de_energy - woa_energy
    )

    pso_vs_woa = (
        pso_energy - woa_energy
    )

    de_vs_pso_pct = (
        de_vs_pso / pso_energy
    ) * 100

    de_vs_woa_pct = (
        de_vs_woa / woa_energy
    ) * 100

    pso_vs_woa_pct = (
        pso_vs_woa / woa_energy
    ) * 100

    print(
        f"DE vs PSO  : "
        f"{de_vs_pso:.4f} MWh "
        f"({de_vs_pso_pct:.4f}%)"
    )

    print(
        f"DE vs WOA  : "
        f"{de_vs_woa:.4f} MWh "
        f"({de_vs_woa_pct:.4f}%)"
    )

    print(
        f"PSO vs WOA : "
        f"{pso_vs_woa:.4f} MWh "
        f"({pso_vs_woa_pct:.4f}%)"
    )

    # ========================================================
    # 9. RUNTIME COMPARISON
    # ========================================================

    de_time = float(
        de["execution_time_seconds"]
    )

    pso_time = float(
        pso["execution_time_seconds"]
    )

    woa_time = float(
        woa["execution_time_seconds"]
    )

    print("\n" + "=" * 70)
    print("RUNTIME COMPARISON")
    print("=" * 70)

    print(
        f"DE  : {de_time:.4f} seconds"
    )

    print(
        f"PSO : {pso_time:.4f} seconds"
    )

    print(
        f"WOA : {woa_time:.4f} seconds"
    )

    print(
        f"\nDE vs PSO runtime difference : "
        f"{de_time - pso_time:.4f} seconds"
    )

    print(
        f"DE vs WOA runtime difference : "
        f"{de_time - woa_time:.4f} seconds"
    )

    # ========================================================
    # 10. FEASIBILITY
    # ========================================================

    print("\n" + "=" * 70)
    print("FEASIBILITY CHECK")
    print("=" * 70)

    for _, row in comparison.iterrows():

        print(
            f"{row['algorithm']} feasible: "
            f"{row['feasible']}"
        )

    # ========================================================
    # 11. CONVERGENCE PLOT
    # ========================================================

    print("\nCreating convergence plot...")

    de_conv = pd.read_csv(
        DE_CONVERGENCE
    )

    pso_conv = pd.read_csv(
        PSO_CONVERGENCE
    )

    woa_conv = pd.read_csv(
        WOA_CONVERGENCE
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        de_conv["iteration"],
        de_conv["best_fitness"],
        label="DE"
    )

    plt.plot(
        pso_conv["iteration"],
        pso_conv["best_fitness"],
        label="PSO"
    )

    plt.plot(
        woa_conv["iteration"],
        woa_conv["best_fitness"],
        label="WOA"
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Best Fitness"
    )

    plt.title(
        "Convergence Comparison of DE, PSO and WOA"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        CONVERGENCE_PLOT,
        dpi=300
    )

    plt.close()

    # ========================================================
    # 12. ENERGY BAR CHART
    # ========================================================

    print("Creating energy comparison plot...")

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        comparison["algorithm"],
        comparison["total_energy_mwh"]
    )

    plt.xlabel(
        "Optimization Algorithm"
    )

    plt.ylabel(
        "Total Energy (MWh)"
    )

    plt.title(
        "Hydropower Energy Generation Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        ENERGY_PLOT,
        dpi=300
    )

    plt.close()

    # ========================================================
    # 13. MONTHLY COMPARISON
    # ========================================================

    print("Creating monthly comparison...")

    de_result = pd.read_csv(
        DE_RESULT
    )

    pso_result = pd.read_csv(
        PSO_RESULT
    )

    woa_result = pd.read_csv(
        WOA_RESULT
    )

    monthly = pd.DataFrame()

    monthly["date"] = de_result["date"]

    monthly[
        "predicted_inflow_m3_s"
    ] = de_result[
        "predicted_inflow_m3_s"
    ]

    monthly[
        "DE_release_m3_s"
    ] = de_result[
        "optimal_release_m3_s"
    ]

    monthly[
        "PSO_release_m3_s"
    ] = pso_result[
        "optimal_release_m3_s"
    ]

    monthly[
        "WOA_release_m3_s"
    ] = woa_result[
        "optimal_release_m3_s"
    ]

    monthly[
        "DE_power_mw"
    ] = de_result[
        "power_mw"
    ]

    monthly[
        "PSO_power_mw"
    ] = pso_result[
        "power_mw"
    ]

    monthly[
        "WOA_power_mw"
    ] = woa_result[
        "power_mw"
    ]

    monthly[
        "DE_energy_mwh"
    ] = de_result[
        "energy_mwh"
    ]

    monthly[
        "PSO_energy_mwh"
    ] = pso_result[
        "energy_mwh"
    ]

    monthly[
        "WOA_energy_mwh"
    ] = woa_result[
        "energy_mwh"
    ]

    monthly.to_csv(
        MONTHLY_COMPARISON_PATH,
        index=False
    )

    # ========================================================
    # 14. FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)

    print(
        f"\nComparison CSV:\n"
        f"{COMPARISON_PATH}"
    )

    print(
        f"\nConvergence plot:\n"
        f"{CONVERGENCE_PLOT}"
    )

    print(
        f"\nEnergy comparison plot:\n"
        f"{ENERGY_PLOT}"
    )

    print(
        f"\nMonthly comparison CSV:\n"
        f"{MONTHLY_COMPARISON_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()