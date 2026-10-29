import pandas as pd
import numpy as np
from pathlib import Path
import time


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = (
    BASE_DIR
    / "results"
    / "forecasts"
    / "long_term_15month_forecast.csv"
)

PARAMETER_FILE = (
    BASE_DIR
    / "data"
    / "hydropower_parameters.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "optimization"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONSTANTS
# ============================================================

RHO_WATER = 1000.0
GRAVITY = 9.81

SECONDS_PER_DAY = 24 * 60 * 60
HOURS_PER_DAY = 24


# ============================================================
# 3. LOAD FORECAST DATA
# ============================================================

print("Loading predicted inflow...")

forecast_df = pd.read_csv(
    FORECAST_FILE
)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)

predicted_inflow = (
    forecast_df["predicted_inflow_m3_s"]
    .astype(float)
    .values
)

NUM_MONTHS = len(
    predicted_inflow
)

days_in_month = (
    forecast_df["date"]
    .dt.days_in_month
    .values
)

print(
    f"Optimization horizon: "
    f"{NUM_MONTHS} months"
)


# ============================================================
# 4. LOAD HYDROPOWER PARAMETERS
# ============================================================

print("\nLoading hydropower parameters...")

parameter_df = pd.read_csv(
    PARAMETER_FILE
)

plant = parameter_df.iloc[0]

HEAD_M = float(
    plant["head_m"]
)

EFFICIENCY = float(
    plant["efficiency"]
)

MIN_TURBINE_FLOW = float(
    plant["min_turbine_flow_m3_s"]
)

MAX_TURBINE_FLOW = float(
    plant["max_turbine_flow_m3_s"]
)

MIN_RELEASE = float(
    plant["min_release_m3_s"]
)

MAX_RELEASE = float(
    plant["max_release_m3_s"]
)

INITIAL_STORAGE = float(
    plant["initial_storage_m3"]
)

MIN_STORAGE = float(
    plant["min_storage_m3"]
)

MAX_STORAGE = float(
    plant["max_storage_m3"]
)


# ============================================================
# 5. HYDROPOWER CALCULATION
# ============================================================

def calculate_power_mw(
    turbine_flow
):

    power_watts = (
        EFFICIENCY
        * RHO_WATER
        * GRAVITY
        * turbine_flow
        * HEAD_M
    )

    return (
        power_watts
        / 1_000_000
    )


# ============================================================
# 6. CONSTRAINT-AWARE RELEASE
# ============================================================

def make_feasible_release(
    proposed_release,
    current_storage,
    inflow,
    days
):

    seconds = (
        days
        * SECONDS_PER_DAY
    )

    inflow_volume = (
        inflow
        * seconds
    )

    storage_based_max_release = (
        current_storage
        + inflow_volume
        - MIN_STORAGE
    ) / seconds

    feasible_upper_bound = min(
        MAX_RELEASE,
        storage_based_max_release
    )

    if (
        feasible_upper_bound
        < MIN_RELEASE
    ):
        return MIN_RELEASE

    feasible_release = np.clip(
        proposed_release,
        MIN_RELEASE,
        feasible_upper_bound
    )

    return feasible_release


# ============================================================
# 7. RESERVOIR SIMULATION
# ============================================================

def simulate_reservoir(
    proposed_releases,
    inflows
):

    storage = np.zeros(
        NUM_MONTHS + 1
    )

    releases = np.zeros(
        NUM_MONTHS
    )

    spill = np.zeros(
        NUM_MONTHS
    )

    storage[0] = (
        INITIAL_STORAGE
    )

    for t in range(
        NUM_MONTHS
    ):

        releases[t] = (
            make_feasible_release(
                proposed_releases[t],
                storage[t],
                inflows[t],
                days_in_month[t]
            )
        )

        seconds = (
            days_in_month[t]
            * SECONDS_PER_DAY
        )

        inflow_volume = (
            inflows[t]
            * seconds
        )

        release_volume = (
            releases[t]
            * seconds
        )

        tentative_storage = (
            storage[t]
            + inflow_volume
            - release_volume
        )

        # ----------------------------------------------------
        # Upper storage constraint
        # Excess water becomes spill
        # ----------------------------------------------------

        if (
            tentative_storage
            > MAX_STORAGE
        ):

            spill[t] = (
                tentative_storage
                - MAX_STORAGE
            )

            storage[t + 1] = (
                MAX_STORAGE
            )

        else:

            storage[t + 1] = (
                tentative_storage
            )

    return (
        storage,
        releases,
        spill
    )


# ============================================================
# 8. ENERGY CALCULATION
# ============================================================

def calculate_energy(
    releases
):

    monthly_power = np.zeros(
        NUM_MONTHS
    )

    monthly_energy = np.zeros(
        NUM_MONTHS
    )

    for t in range(
        NUM_MONTHS
    ):

        turbine_flow = np.clip(
            releases[t],
            MIN_TURBINE_FLOW,
            MAX_TURBINE_FLOW
        )

        monthly_power[t] = (
            calculate_power_mw(
                turbine_flow
            )
        )

        monthly_energy[t] = (
            monthly_power[t]
            * days_in_month[t]
            * HOURS_PER_DAY
        )

    total_energy = np.sum(
        monthly_energy
    )

    return (
        total_energy,
        monthly_power,
        monthly_energy
    )


# ============================================================
# 9. FITNESS FUNCTION
# ============================================================

def fitness_function(
    proposed_releases,
    inflows
):

    (
        storage,
        feasible_releases,
        spill
    ) = simulate_reservoir(
        proposed_releases,
        inflows
    )

    (
        total_energy,
        _,
        _
    ) = calculate_energy(
        feasible_releases
    )

    # --------------------------------------------------------
    # Storage constraint penalty
    # --------------------------------------------------------

    storage_violation = np.maximum(
        MIN_STORAGE
        - storage[1:],
        0
    )

    storage_penalty = np.sum(
        (
            storage_violation
            / 1_000_000
        ) ** 2
    )

    # --------------------------------------------------------
    # Spill penalty
    # --------------------------------------------------------

    spill_penalty = (
        np.sum(spill)
        / 1_000_000
    ) ** 2

    PENALTY_STORAGE = 10_000_000.0
    PENALTY_SPILL = 1.0

    penalty = (
        PENALTY_STORAGE
        * storage_penalty
        +
        PENALTY_SPILL
        * spill_penalty
    )

    # --------------------------------------------------------
    # DE minimizes fitness
    # --------------------------------------------------------

    fitness = (
        -total_energy
        + penalty
    )

    return fitness


# ============================================================
# 10. DIFFERENTIAL EVOLUTION PARAMETERS
# ============================================================

POPULATION_SIZE = 40

MAX_ITERATIONS = 150

# Differential Evolution parameters
MUTATION_FACTOR = 0.8

CROSSOVER_RATE = 0.9


# ============================================================
# 11. DIFFERENTIAL EVOLUTION
# ============================================================

def run_de(
    seed=42,
    save_results=True
):

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    np.random.seed(seed)

    dimensions = NUM_MONTHS

    lower_bound = np.full(
        dimensions,
        MIN_RELEASE
    )

    upper_bound = np.full(
        dimensions,
        MAX_RELEASE
    )

    # --------------------------------------------------------
    # Initialize population
    # --------------------------------------------------------

    population = np.random.uniform(
        lower_bound,
        upper_bound,
        size=(
            POPULATION_SIZE,
            dimensions
        )
    )

    # --------------------------------------------------------
    # Initial fitness
    # --------------------------------------------------------

    fitness_values = np.array([

        fitness_function(
            individual,
            predicted_inflow
        )

        for individual in population

    ])

    best_index = np.argmin(
        fitness_values
    )

    best_position = (
        population[
            best_index
        ].copy()
    )

    best_fitness = (
        fitness_values[
            best_index
        ]
    )

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.time()

    convergence_history = []

    # ========================================================
    # DE MAIN LOOP
    # ========================================================

    for iteration in range(
        MAX_ITERATIONS
    ):

        for i in range(
            POPULATION_SIZE
        ):

            # ------------------------------------------------
            # Select three different individuals
            # ------------------------------------------------

            candidates = [
                j
                for j in range(
                    POPULATION_SIZE
                )
                if j != i
            ]

            r1, r2, r3 = np.random.choice(
                candidates,
                3,
                replace=False
            )

            # ------------------------------------------------
            # Mutation
            #
            # v = x1 + F(x2 - x3)
            # ------------------------------------------------

            mutant = (
                population[r1]
                +
                MUTATION_FACTOR
                * (
                    population[r2]
                    -
                    population[r3]
                )
            )

            # ------------------------------------------------
            # Apply bounds
            # ------------------------------------------------

            mutant = np.clip(
                mutant,
                lower_bound,
                upper_bound
            )

            # ------------------------------------------------
            # Binomial crossover
            # ------------------------------------------------

            trial = population[i].copy()

            crossover_mask = (
                np.random.random(
                    dimensions
                )
                < CROSSOVER_RATE
            )

            # Ensure at least one mutant dimension
            forced_index = np.random.randint(
                dimensions
            )

            crossover_mask[
                forced_index
            ] = True

            trial[
                crossover_mask
            ] = mutant[
                crossover_mask
            ]

            # ------------------------------------------------
            # Evaluate trial solution
            # ------------------------------------------------

            trial_fitness = (
                fitness_function(
                    trial,
                    predicted_inflow
                )
            )

            # ------------------------------------------------
            # Greedy selection
            # ------------------------------------------------

            if (
                trial_fitness
                <
                fitness_values[i]
            ):

                population[i] = (
                    trial.copy()
                )

                fitness_values[i] = (
                    trial_fitness
                )

            # ------------------------------------------------
            # Update global best
            # ------------------------------------------------

            if (
                fitness_values[i]
                <
                best_fitness
            ):

                best_fitness = (
                    fitness_values[i]
                )

                best_position = (
                    population[i].copy()
                )

        # ----------------------------------------------------
        # Store convergence
        # ----------------------------------------------------

        convergence_history.append(
            best_fitness
        )
        if (
    (iteration + 1) % 10 == 0
    or iteration == 0):
            print(
            f"DE Iteration "
            f"{iteration + 1}/{MAX_ITERATIONS} | "
            f"Best Fitness: "
            f"{best_fitness:.4f}"
    )

    # ========================================================
    # EXECUTION TIME
    # ========================================================

    execution_time = (
        time.time()
        - start_time
    )

    # ========================================================
    # FINAL SOLUTION
    # ========================================================

    (
        final_storage,
        final_releases,
        final_spill
    ) = simulate_reservoir(
        best_position,
        predicted_inflow
    )

    (
        total_energy,
        monthly_power,
        monthly_energy
    ) = calculate_energy(
        final_releases
    )

    # ========================================================
    # CONSTRAINT CHECK
    # ========================================================

    storage_violation = np.sum(
        np.maximum(
            MIN_STORAGE
            - final_storage[1:],
            0
        )
    )

    minimum_storage = np.min(
        final_storage
    )

    maximum_storage = np.max(
        final_storage
    )

    total_spill = np.sum(
        final_spill
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if save_results:

        result_df = pd.DataFrame({

            "date":
                forecast_df["date"],

            "predicted_inflow_m3_s":
                predicted_inflow,

            "proposed_release_m3_s":
                best_position,

            "optimal_release_m3_s":
                final_releases,

            "storage_start_m3":
                final_storage[:-1],

            "storage_end_m3":
                final_storage[1:],

            "spill_m3":
                final_spill,

            "power_mw":
                monthly_power,

            "energy_mwh":
                monthly_energy

        })

        result_file = (
            RESULTS_DIR
            / "de_hydropower_optimization.csv"
        )

        result_df.to_csv(
            result_file,
            index=False
        )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        convergence_df = pd.DataFrame({

            "iteration":
                np.arange(
                    1,
                    MAX_ITERATIONS + 1
                ),

            "best_fitness":
                convergence_history

        })

        convergence_file = (
            RESULTS_DIR
            / "de_convergence.csv"
        )

        convergence_df.to_csv(
            convergence_file,
            index=False
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_df = pd.DataFrame([{

            "algorithm":
                "DE",

            "seed":
                seed,

            "best_fitness":
                best_fitness,

            "total_energy_mwh":
                total_energy,

            "total_spill_m3":
                total_spill,

            "storage_violation_m3":
                storage_violation,

            "minimum_storage_m3":
                minimum_storage,

            "maximum_storage_m3":
                maximum_storage,

            "execution_time_seconds":
                execution_time,

            "iterations":
                MAX_ITERATIONS,

            "population_size":
                POPULATION_SIZE

        }])

        summary_file = (
            RESULTS_DIR
            / "de_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
    "algorithm": "DE",
    "seed": seed,
    "best_fitness": best_fitness,
    "total_energy_mwh": total_energy,
    "total_spill_m3": total_spill,
    "storage_violation_m3": storage_violation,
    "minimum_storage_m3": minimum_storage,
    "maximum_storage_m3": maximum_storage,
    "execution_time_seconds": execution_time,
    "iterations": MAX_ITERATIONS,
    "population_size": POPULATION_SIZE,
    "convergence": convergence_history
}

# ============================================================
# 12. NORMAL SINGLE RUN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("STARTING CONSTRAINT-AWARE DE")
    print("========================================")

    result = run_de(
        seed=42,
        save_results=True
    )

    print("\n")
    print("=" * 60)
    print("CONSTRAINT-AWARE DE COMPLETED")
    print("=" * 60)

    print(
        f"Best fitness        : "
        f"{result['best_fitness']:.4f}"
    )

    print(
        f"Total energy        : "
        f"{result['total_energy_mwh']:.4f} MWh"
    )

    print(
        f"Total spill         : "
        f"{result['total_spill_m3']:.4f} m3"
    )

    print(
        f"Storage violation   : "
        f"{result['storage_violation_m3']:.4f} m3"
    )

    print(
        f"Minimum storage     : "
        f"{result['minimum_storage_m3']:.4f} m3"
    )

    print(
        f"Maximum storage     : "
        f"{result['maximum_storage_m3']:.4f} m3"
    )

    print(
        f"Execution time      : "
        f"{result['execution_time_seconds']:.4f} seconds"
    )