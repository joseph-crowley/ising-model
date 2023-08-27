import numpy as np
import time
from utils import renormalize, energy_vectorized, calculate_nn_sums

def monte_carlo_step_optimized(spins, nn_sums, beta, J, H, L_current):
    """
    Perform a Monte Carlo step for the Ising model simulation.

    Args:
        spins (np.ndarray): 2D array representing the spins of the lattice.
        nn_sums (np.ndarray): 2D array representing the nearest-neighbor sums.
        beta (float): 1 / temperature.
        J (float): Interaction strength.
        H (float): External magnetic field strength.
        L_current (int): The current size of the lattice.

    Returns:
        None: The function modifies the spins and nn_sums arrays in-place.
    """
    for _ in range(L_current * L_current):
        i, j = np.random.randint(0, L_current, 2)
        dE = 2 * J * spins[i, j] * nn_sums[i, j] + 2 * H * spins[i, j]
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                i_nn, j_nn = (i + dx) % L_current, (j + dy) % L_current
                nn_sums[i_nn, j_nn] += 2 * spins[i, j]


def simulate_for_temperature(T, L, J, H, n_steps, n_measure):
    """
    Simulate the Ising model for a single temperature.

    Args:
        T (float): Temperature.
        L (int): Initial size of the lattice.
        J (float): Interaction strength.
        H (float): External magnetic field strength.
        n_steps (int): Number of equilibration steps.
        n_measure (int): Number of measurement steps.

    Returns:
        tuple: A tuple containing temperature, average magnetization, susceptibility, and other metrics.
    """
    print(f"Starting simulation for T = {T:.2f}...")
    start_time_sim = time.time()

    # Initialize spins and nearest-neighbor sums
    spins = np.ones((L, L))
    nn_sums = calculate_nn_sums(spins)
    beta = 1 / T

    M_vals = []
    L_current = L
    
    # Equilibration steps
    for _ in range(n_steps):
        monte_carlo_step_optimized(spins, nn_sums, beta, J, H, L_current)
        
    # Measurement steps
    """
    The idea is that the system first undergoes a "burn-in" period of N_STEPS to reach equilibrium, after which measurements are taken over N_MEASURE steps to calculate average values and variances for the observables of interest.
    """
    for _ in range(n_measure):
        monte_carlo_step_optimized(spins, nn_sums, beta, J, H, L_current)
        M = np.mean(spins)
        M_vals.append(M)
        
    M_avg = np.mean(M_vals)
    M_var = np.var(M_vals)
    chi = beta * M_var * L_current * L_current

    # Renormalization
    spins_renormalized, L_renormalized = renormalize(spins, L_current)
    nn_sums_renormalized = calculate_nn_sums(spins_renormalized)
    M_vals_renormalized = []
    for _ in range(n_measure):
        monte_carlo_step_optimized(spins_renormalized, nn_sums_renormalized, beta, J, H, L_renormalized)
        M = np.mean(spins_renormalized)
        M_vals_renormalized.append(M)
        
    M_avg_renormalized = np.mean(M_vals_renormalized)
    M_var_renormalized = np.var(M_vals_renormalized)
    chi_renormalized = beta * M_var_renormalized * L_renormalized * L_renormalized

    elapsed_time_sim = time.time() - start_time_sim
    print(f"Simulation for T = {T:.2f} completed in {elapsed_time_sim:.2f} seconds.")
    
    return T, M_avg, chi, M_avg_renormalized, chi_renormalized, spins, spins_renormalized, M_vals
