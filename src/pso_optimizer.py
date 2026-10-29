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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

forecast_df = pd.read_csv(FORECAST_FILE)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)

predicted_inflow = (
    forecast_df["predicted_inflow_m3_s"]
    .astype(float)
    .values
)

NUM_MONTHS = len(predicted_inflow)

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

def calculate_power_mw(turbine_flow):

    power_watts = (
        EFFICIENCY
        * RHO_WATER
        * GRAVITY
        * turbine_flow
        * HEAD_M
    )

    return power_watts / 1_000_000


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
        days * SECONDS_PER_DAY
    )

    inflow_volume = (
        inflow * seconds
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

    if feasible_upper_bound < MIN_RELEASE:

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

    storage[0] = INITIAL_STORAGE

    for t in range(NUM_MONTHS):

        releases[t] = make_feasible_release(
            proposed_releases[t],
            storage[t],
            inflows[t],
            days_in_month[t]
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

        if tentative_storage > MAX_STORAGE:

            spill[t] = (
                tentative_storage
                - MAX_STORAGE
            )

            storage[t + 1] = MAX_STORAGE

        else:

            storage[t + 1] = tentative_storage

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

    for t in range(NUM_MONTHS):

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

    fitness = (
        -total_energy
        + penalty
    )

    return fitness


# ============================================================
# 10. PSO PARAMETERS
# ============================================================

NUM_PARTICLES = 40
MAX_ITERATIONS = 150

INERTIA_WEIGHT = 0.7
COGNITIVE_COEFFICIENT = 1.5
SOCIAL_COEFFICIENT = 1.5


# ============================================================
# 11. PSO FUNCTION
# ============================================================

def run_pso(
    seed=42,
    save_results=True
):

    # --------------------------------------------------------
    # Set seed for this particular experiment
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
    # Initialize particles
    # --------------------------------------------------------

    positions = np.random.uniform(
        lower_bound,
        upper_bound,
        size=(
            NUM_PARTICLES,
            dimensions
        )
    )

    velocities = np.zeros_like(
        positions
    )

    # --------------------------------------------------------
    # Initial fitness
    # --------------------------------------------------------

    personal_best_positions = (
        positions.copy()
    )

    personal_best_fitness = np.array([

        fitness_function(
            particle,
            predicted_inflow
        )

        for particle in positions

    ])

    best_index = np.argmin(
        personal_best_fitness
    )

    global_best_position = (
        personal_best_positions[
            best_index
        ].copy()
    )

    global_best_fitness = (
        personal_best_fitness[
            best_index
        ]
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    start_time = time.time()

    # IMPORTANT:
    # This list is returned to optimization_multi_run.py
    convergence_history = []

    for iteration in range(
        MAX_ITERATIONS
    ):

        for i in range(
            NUM_PARTICLES
        ):

            r1 = np.random.random(
                dimensions
            )

            r2 = np.random.random(
                dimensions
            )

            velocities[i] = (

                INERTIA_WEIGHT
                * velocities[i]

                +

                COGNITIVE_COEFFICIENT
                * r1
                * (
                    personal_best_positions[i]
                    - positions[i]
                )

                +

                SOCIAL_COEFFICIENT
                * r2
                * (
                    global_best_position
                    - positions[i]
                )
            )

            positions[i] = (
                positions[i]
                + velocities[i]
            )

            positions[i] = np.clip(
                positions[i],
                lower_bound,
                upper_bound
            )

            current_fitness = (
                fitness_function(
                    positions[i],
                    predicted_inflow
                )
            )

            if (
                current_fitness
                <
                personal_best_fitness[i]
            ):

                personal_best_fitness[i] = (
                    current_fitness
                )

                personal_best_positions[i] = (
                    positions[i].copy()
                )

            if (
                current_fitness
                <
                global_best_fitness
            ):

                global_best_fitness = (
                    current_fitness
                )

                global_best_position = (
                    positions[i].copy()
                )

        # Store best fitness after every iteration
        convergence_history.append(
            global_best_fitness
        )
        if (
    (iteration + 1) % 10 == 0
    or iteration == 0):
            print(
            f"PSO Iteration "
            f"{iteration + 1}/{MAX_ITERATIONS} | "
            f"Best Fitness: "
            f"{global_best_fitness:.4f}"
    )

    execution_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Final solution
    # --------------------------------------------------------

    (
        final_storage,
        final_releases,
        final_spill
    ) = simulate_reservoir(
        global_best_position,
        predicted_inflow
    )

    (
        total_energy,
        monthly_power,
        monthly_energy
    ) = calculate_energy(
        final_releases
    )

    # --------------------------------------------------------
    # Constraint check
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if save_results:

        result_df = pd.DataFrame({

            "date":
                forecast_df["date"],

            "predicted_inflow_m3_s":
                predicted_inflow,

            "proposed_release_m3_s":
                global_best_position,

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
            / "pso_hydropower_optimization.csv"
        )

        result_df.to_csv(
            result_file,
            index=False
        )

        # ----------------------------------------------------
        # Save convergence history
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
            / "pso_convergence.csv"
        )

        convergence_df.to_csv(
            convergence_file,
            index=False
        )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        summary_df = pd.DataFrame([{

            "algorithm":
                "PSO",

            "seed":
                seed,

            "best_fitness":
                global_best_fitness,

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
                NUM_PARTICLES

        }])

        summary_file = (
            RESULTS_DIR
            / "pso_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

    # --------------------------------------------------------
    # Return results for multi-run experiment
    # --------------------------------------------------------

    return {

        "algorithm":
            "PSO",

        "seed":
            seed,

        "best_fitness":
            global_best_fitness,

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
            NUM_PARTICLES,

        # IMPORTANT:
        # Required by optimization_multi_run.py
        "convergence":
            convergence_history
    }


# ============================================================
# 12. NORMAL SINGLE RUN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("STARTING CONSTRAINT-AWARE PSO")
    print("========================================")

    result = run_pso(
        seed=42,
        save_results=True
    )

    print("\n")
    print("=" * 60)
    print("CONSTRAINT-AWARE PSO COMPLETED")
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

    print(
        f"Convergence points  : "
        f"{len(result['convergence'])}"
    )