import numpy as np
import scipy.optimize as opt

def renormalize(spins, L_current):
    """
    Renormalize the spin lattice by reducing its size and adjusting the spins.

    Args:
        spins (np.ndarray): 2D array representing the spins of the lattice.
        L_current (int): The current size of the lattice.

    Returns:
        tuple: A tuple containing the renormalized spins (np.ndarray) and the new lattice size (int).
    """
    L_new = L_current // 2
    spins_new = np.zeros((L_new, L_new))
    for i in range(L_new):
        for j in range(L_new):
            spins_new[i, j] = np.sign(np.sum(spins[2*i:2*(i+1), 2*j:2*(j+1)]))
    return spins_new, L_new

def energy_vectorized(spins, J, H):
    """
    Calculate the energy of the spin lattice.

    Args:
        spins (np.ndarray): 2D array representing the spins of the lattice.
        J (float): The interaction strength parameter.
        H (float): The external magnetic field strength.

    Returns:
        float: The energy of the lattice.
    """
    E = -J * np.sum(
        spins * (
            np.roll(spins, shift=-1, axis=0) + 
            np.roll(spins, shift=-1, axis=1)
        )
    ) - H * np.sum(spins)
    return E


def calculate_nn_sums(spins):
    """
    Calculate the nearest-neighbor sums for each spin in the lattice.

    Args:
        spins (np.ndarray): 2D array representing the spins of the lattice.

    Returns:
        np.ndarray: 2D array containing the nearest-neighbor sums for each spin.
    """
    nn_sums = (
        np.roll(spins, shift=-1, axis=0) + 
        np.roll(spins, shift=1, axis=0) +
        np.roll(spins, shift=-1, axis=1) + 
        np.roll(spins, shift=1, axis=1)
    )
    return nn_sums

def exponential_decay(r, xi):
    """
    Exponential decay function for curve fitting.

    Args:
        r (np.ndarray): The distance array.
        xi (float): The correlation length.

    Returns:
        np.ndarray: The exponential decay values.
    """
    return np.exp(-r / xi)

def calculate_correlation_length(spins, L):
    """
    Calculate the correlation length of the spin lattice.

    Args:
        spins (np.ndarray): 2D array representing the spins of the lattice.
        L (int): The size of the lattice.

    Returns:
        float: The correlation length.
    """
    G = np.zeros(L//4)
    for dx in range(1, L//4):
        for dy in range(1, L//4):
            G[dx] += np.mean(spins * np.roll(spins, shift=(dx, dy), axis=(0, 1)))
    G -= np.mean(spins) ** 2
    params, _ = opt.curve_fit(exponential_decay, np.arange(1, L//4), G[1:])
    xi = params[0]
    return xi

def calculate_binder_cumulant(M_vals):
    """
    Calculate the Binder cumulant of the spin lattice.

    Args:
        M_vals (list): List of magnetization values.

    Returns:
        float: The Binder cumulant.
    """
    M2 = np.mean(np.array(M_vals) ** 2)
    M4 = np.mean(np.array(M_vals) ** 4)
    U = 1 - M4 / (3 * M2 ** 2)
    return U
