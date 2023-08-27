# Constants used in the simulation
import numpy as np
L = 32  # Lattice size (should be a power of 2 for easier renormalization)
J = 1.0  # Interaction strength
H = 0.0  # External magnetic field strength
N_STEPS = 10000  # Number of Monte Carlo steps for equilibration
N_MEASURE = 100  # Number of additional Monte Carlo steps for measurement
T_VALUES = np.arange(2.3,2.8,.05)  # Range of temperatures
