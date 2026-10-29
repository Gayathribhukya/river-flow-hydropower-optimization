import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pso_optimizer import run_pso
from woa_optimizer import run_woa
from de_optimizer import run_de


# ============================================================
# SETTINGS
# ============================================================

SEEDS = [42, 43, 44, 45, 46]

RESULTS_DIR = "results/optimization"
PLOTS_DIR = "results/plots"

RUN_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "pso_woa_de_multi_run_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "pso_woa_de_multi_run_summary.csv"
)

ENERGY_PLOT = os.path.join(
    PLOTS_DIR,
    "pso_woa_de_runwise_energy.png"
)

RUNTIME_PLOT = os.path.join(
    PLOTS_DIR,
    "pso_woa_de_runwise_runtime.png"
)

MEAN_ENERGY_PLOT = os.path.join(
    PLOTS_DIR,
    "pso_woa_de_mean_energy.png"
)

CONVERGENCE_PLOT = os.path.join(
    PLOTS_DIR,
    "pso_woa_de_convergence.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    print("=" * 75)
    print("5-RUN PSO vs WOA vs DE COMPARISON")
    print("=" * 75)

    all_results = []

    convergence_data = {
        "PSO": [],
        "WOA": [],
        "DE": []
    }

    # ========================================================
    # RUN ALL THREE ALGORITHMS
    # ========================================================

    for run_number, seed in enumerate(SEEDS, start=1):

        print("\n")
        print("-" * 75)
        print(
            f"RUN {run_number} / {len(SEEDS)}"
        )
        print(
            f"Seed: {seed}"
        )
        print("-" * 75)

        # ----------------------------------------------------
        # PSO
        # ----------------------------------------------------

        print("\nRunning PSO...")

        pso_result = run_pso(
            seed=seed,
            save_results=False
        )

        all_results.append({
            "run": run_number,
            "seed": seed,
            "algorithm": "PSO",
            "total_energy_mwh":
                pso_result["total_energy_mwh"],
            "best_fitness":
                pso_result["best_fitness"],
            "execution_time_seconds":
                pso_result["execution_time_seconds"],
            "total_spill_m3":
                pso_result["total_spill_m3"],
            "storage_violation_m3":
                pso_result["storage_violation_m3"],
            
        })

        convergence_data["PSO"].append(
            pso_result["convergence"]
        )

        # ----------------------------------------------------
        # WOA
        # ----------------------------------------------------

        print("\nRunning WOA...")

        woa_result = run_woa(
            seed=seed,
            save_results=False
        )

        all_results.append({
            "run": run_number,
            "seed": seed,
            "algorithm": "WOA",
            "total_energy_mwh":
                woa_result["total_energy_mwh"],
            "best_fitness":
                woa_result["best_fitness"],
            "execution_time_seconds":
                woa_result["execution_time_seconds"],
            "total_spill_m3":
                woa_result["total_spill_m3"],
            "storage_violation_m3":
                woa_result["storage_violation_m3"],
            
        })

        convergence_data["WOA"].append(
            woa_result["convergence"]
        )

        # ----------------------------------------------------
        # DE
        # ----------------------------------------------------

        print("\nRunning DE...")

        de_result = run_de(
            seed=seed,
            save_results=False
        )

        all_results.append({
            "run": run_number,
            "seed": seed,
            "algorithm": "DE",
            "total_energy_mwh":
                de_result["total_energy_mwh"],
            "best_fitness":
                de_result["best_fitness"],
            "execution_time_seconds":
                de_result["execution_time_seconds"],
            "total_spill_m3":
                de_result["total_spill_m3"],
            "storage_violation_m3":
                de_result["storage_violation_m3"],
            
        })

        convergence_data["DE"].append(
            de_result["convergence"]
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # FEASIBILITY
    # ========================================================

    TOLERANCE = 1e-6

    results_df["feasible"] = (
        results_df[
            "storage_violation_m3"
        ].abs()
        <= TOLERANCE
    )

    # ========================================================
    # SAVE RAW RUN RESULTS
    # ========================================================

    results_df.to_csv(
        RUN_RESULTS_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for algorithm in [
        "DE",
        "PSO",
        "WOA"
    ]:

        df = results_df[
            results_df["algorithm"]
            == algorithm
        ]

        summary_rows.append({
            "algorithm":
                algorithm,

            "mean_energy_mwh":
                df[
                    "total_energy_mwh"
                ].mean(),

            "std_energy_mwh":
                df[
                    "total_energy_mwh"
                ].std(),

            "best_energy_mwh":
                df[
                    "total_energy_mwh"
                ].max(),

            "worst_energy_mwh":
                df[
                    "total_energy_mwh"
                ].min(),

            "mean_fitness":
                df[
                    "best_fitness"
                ].mean(),

            "mean_runtime_seconds":
                df[
                    "execution_time_seconds"
                ].mean(),

            "std_runtime_seconds":
                df[
                    "execution_time_seconds"
                ].std(),

            "best_runtime_seconds":
                df[
                    "execution_time_seconds"
                ].min(),

            "worst_runtime_seconds":
                df[
                    "execution_time_seconds"
                ].max(),

            "mean_spill_m3":
                df[
                    "total_spill_m3"
                ].mean(),

            "max_storage_violation_m3":
                df[
                    "storage_violation_m3"
                ].abs().max(),

            "all_runs_feasible":
                df["feasible"].all()
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # PRINT RUN-WISE RESULTS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("RUN-WISE ENERGY RESULTS")
    print("=" * 75)

    energy_table = results_df[
        [
            "run",
            "seed",
            "algorithm",
            "total_energy_mwh"
        ]
    ]

    print(
        energy_table.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("5-RUN SUMMARY")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # DETERMINE BEST ALGORITHM
    # ========================================================

    best_mean_index = summary_df[
        "mean_energy_mwh"
    ].idxmax()

    best_algorithm = summary_df.loc[
        best_mean_index,
        "algorithm"
    ]

    best_mean_energy = summary_df.loc[
        best_mean_index,
        "mean_energy_mwh"
    ]

    fastest_index = summary_df[
        "mean_runtime_seconds"
    ].idxmin()

    fastest_algorithm = summary_df.loc[
        fastest_index,
        "algorithm"
    ]

    print("\n")
    print("=" * 75)
    print("FINAL 5-RUN CONCLUSION")
    print("=" * 75)

    print(
        f"Best mean energy : "
        f"{best_algorithm}"
    )

    print(
        f"Mean energy      : "
        f"{best_mean_energy:.4f} MWh"
    )

    print(
        f"Fastest mean     : "
        f"{fastest_algorithm}"
    )

    # ========================================================
    # PLOT 1 — RUN-WISE ENERGY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in [
        "PSO",
        "WOA",
        "DE"
    ]:

        df = results_df[
            results_df["algorithm"]
            == algorithm
        ]

        plt.plot(
            df["run"],
            df["total_energy_mwh"],
            marker="o",
            label=algorithm
        )

    plt.xlabel(
        "Run"
    )

    plt.ylabel(
        "Total Energy (MWh)"
    )

    plt.title(
        "Run-wise PSO vs WOA vs DE Energy Comparison"
    )

    plt.xticks(
    range(1, len(SEEDS) + 1)
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        ENERGY_PLOT,
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2 — RUN-WISE RUNTIME
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in [
        "PSO",
        "WOA",
        "DE"
    ]:

        df = results_df[
            results_df["algorithm"]
            == algorithm
        ]

        plt.plot(
            df["run"],
            df["execution_time_seconds"],
            marker="o",
            label=algorithm
        )

    plt.xlabel(
        "Run"
    )

    plt.ylabel(
        "Execution Time (seconds)"
    )

    plt.title(
        "Run-wise PSO vs WOA vs DE Runtime Comparison"
    )

    plt.xticks(
        range(1, len(SEEDS) + 1)
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RUNTIME_PLOT,
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3 — MEAN ENERGY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        summary_df["algorithm"],
        summary_df["mean_energy_mwh"],
        yerr=summary_df["std_energy_mwh"],
        capsize=5
    )

    plt.xlabel(
        "Optimization Algorithm"
    )

    plt.ylabel(
        "Mean Energy (MWh)"
    )

    plt.title(
        "Mean Energy Generation Across 5 Runs"
    )

    plt.tight_layout()

    plt.savefig(
        MEAN_ENERGY_PLOT,
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4 — MEAN CONVERGENCE
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in [
        "PSO",
        "WOA",
        "DE"
    ]:

        convergence_array = np.array(
            convergence_data[algorithm]
        )

        mean_convergence = (
            np.mean(
                convergence_array,
                axis=0
            )
        )

        plt.plot(
            range(
                1,
                len(mean_convergence) + 1
            ),
            mean_convergence,
            label=algorithm
        )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Mean Best Fitness"
    )

    plt.title(
        "Mean Convergence of PSO, WOA and DE Across 5 Runs"
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
    # FINAL FILES
    # ========================================================

    print("\n")
    print("=" * 75)
    print("5-RUN COMPARISON COMPLETED")
    print("=" * 75)

    print(
        f"\nRun results:\n"
        f"{RUN_RESULTS_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nEnergy plot:\n"
        f"{ENERGY_PLOT}"
    )

    print(
        f"\nRuntime plot:\n"
        f"{RUNTIME_PLOT}"
    )

    print(
        f"\nMean energy plot:\n"
        f"{MEAN_ENERGY_PLOT}"
    )

    print(
        f"\nMean convergence plot:\n"
        f"{CONVERGENCE_PLOT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()