import pandas as pd
from pathlib import Path


# ============================================================
# CONSTANTS
# ============================================================

RHO_WATER = 1000.0
GRAVITY = 9.81


# ============================================================
# POWER CALCULATION
# ============================================================

def calculate_power_mw(
    flow_m3_s,
    head_m,
    efficiency
):

    power_watts = (
        efficiency
        * RHO_WATER
        * GRAVITY
        * flow_m3_s
        * head_m
    )

    power_mw = (
        power_watts
        / 1_000_000
    )

    return power_mw


# ============================================================
# ENERGY CALCULATION
# ============================================================

def calculate_energy_mwh(
    power_mw,
    hours
):

    return (
        power_mw
        * hours
    )


# ============================================================
# LOAD HYDROPOWER PARAMETERS
# ============================================================

def load_hydropower_parameters():

    base_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    parameter_file = (
        base_dir
        / "data"
        / "hydropower_parameters.csv"
    )

    parameters = pd.read_csv(
        parameter_file
    )

    return parameters


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    parameters = (
        load_hydropower_parameters()
    )

    print(
        "\nHydropower parameters:"
    )

    print(parameters)

    plant = parameters.iloc[0]

    sample_flow = 50.0

    power = calculate_power_mw(
        flow_m3_s=sample_flow,
        head_m=plant["head_m"],
        efficiency=plant["efficiency"]
    )

    energy = calculate_energy_mwh(
        power,
        24
    )

    print(
        "\nSample turbine flow:",
        sample_flow,
        "m3/s"
    )

    print(
        "Estimated power:",
        round(power, 4),
        "MW"
    )

    print(
        "Estimated energy in 24 hours:",
        round(energy, 4),
        "MWh"
    )