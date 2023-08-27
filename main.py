import time
from concurrent.futures import ProcessPoolExecutor
from constants import L, J, H, N_STEPS, N_MEASURE, T_VALUES
from utils import calculate_correlation_length, calculate_binder_cumulant
from simulation import simulate_for_temperature
from visualization import create_visualizations, create_advanced_visualizations
from functools import partial
import numpy as np

def main_simulation():
    """
    Main function for running the Ising model simulation.

    Returns:
        None: Outputs are visualizations and measurements.
    """
    print(f"Starting main simulation for {len(T_VALUES)} temperatures between {min(T_VALUES):.2f} and {max(T_VALUES):.2f}...")
    print(f'\nSimulation parameters: L = {L}, J = {J}, H = {H}, N_STEPS = {N_STEPS}, N_MEASURE = {N_MEASURE}\n')
    start_time = time.time()

    spin_configs = {}
    spin_configs_renormalized = {}

    # Partial function for running the simulation for a given temperature
    sim_func = partial(simulate_for_temperature, L=L, J=J, H=H, n_steps=N_STEPS, n_measure=N_MEASURE)

    # Running the simulation for different temperatures in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(sim_func, T_VALUES))

    # Sort results in ascending order of temperature
    sorted_results = sorted(results, key=lambda x: x[0])
    
    temperatures, magnetization, susceptibility, magnetization_renormalized, susceptibility_renormalized, spin_configs_list, spin_configs_renormalized_list, M_vals_list = zip(*sorted_results)

    for T, spins, spins_renormalized in zip(temperatures, spin_configs_list, spin_configs_renormalized_list):
        spin_configs[T] = spins
        spin_configs_renormalized[T] = spins_renormalized

    elapsed_time = time.time() - start_time
    print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")

    # Basic visualizations
    create_visualizations(temperatures, magnetization, magnetization_renormalized, susceptibility, susceptibility_renormalized)

    # Advanced analysis and visualizations
    Tc_index = np.argmax(susceptibility)
    Tc = temperatures[Tc_index]
    print(f"Estimated critical temperature: {Tc:.2f}")

    correlation_lengths = [calculate_correlation_length(spin_configs[T], L) for T in temperatures]
    binder_cumulants = [calculate_binder_cumulant(M_vals) for M_vals in M_vals_list]
    
    create_advanced_visualizations(temperatures, magnetization, magnetization_renormalized, Tc, spin_configs, spin_configs_renormalized, correlation_lengths, binder_cumulants)

    print(f'Total runtime including plotting: {time.time() - start_time:.2f} seconds')

if __name__ == '__main__':
    main_simulation()
