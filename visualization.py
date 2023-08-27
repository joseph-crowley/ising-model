import matplotlib.pyplot as plt
import numpy as np

def create_visualizations(temperatures, magnetization, magnetization_renormalized, susceptibility, susceptibility_renormalized):
    """
    Create visualizations for average magnetization and susceptibility.

    Args:
        temperatures (list): List of temperatures used in the simulation.
        magnetization (list): List of average magnetizations for each temperature.
        magnetization_renormalized (list): List of renormalized average magnetizations for each temperature.
        susceptibility (list): List of susceptibilities for each temperature.
        susceptibility_renormalized (list): List of renormalized susceptibilities for each temperature.

    Returns:
        None: Shows the plots.
    """
    plt.figure(figsize=(18, 8))

    # Plot for Average Magnetization vs Temperature
    plt.subplot(1, 2, 1)
    plt.plot(temperatures, magnetization, marker='o', label='Original')
    plt.plot(temperatures, magnetization_renormalized, marker='x', label='Renormalized')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.title('Average Magnetization vs Temperature')
    plt.legend()
    plt.grid(True)
    
    # Plot for Susceptibility vs Temperature
    plt.subplot(1, 2, 2)
    plt.plot(temperatures, susceptibility, marker='o', label='Original')
    plt.plot(temperatures, susceptibility_renormalized, marker='x', label='Renormalized')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.title('Susceptibility vs Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('magnetization_susceptibility.png')

def create_advanced_visualizations(temperatures, magnetization, magnetization_renormalized, Tc, spin_configs, spin_configs_renormalized, correlation_lengths, binder_cumulants):
    """
    Create advanced visualizations including flow of fixed points under renormalization,
    log-log plot for the order parameter near Tc, spin configurations at Tc,
    correlation length, and Binder cumulant.

    Args:
        temperatures (list): List of temperatures used in the simulation.
        magnetization (list): List of average magnetizations for each temperature.
        magnetization_renormalized (list): List of renormalized average magnetizations for each temperature.
        Tc (float): Estimated critical temperature.
        spin_configs (dict): Dictionary mapping temperatures to spin configurations.
        spin_configs_renormalized (dict): Dictionary mapping temperatures to renormalized spin configurations.
        correlation_lengths (list): List of correlation lengths for each temperature.
        binder_cumulants (list): List of Binder cumulants for each temperature.

    Returns:
        None: Saves the plots as PNG files.
    """
    # Flow of Fixed Points under Renormalization
    plt.figure()
    plt.plot(temperatures, magnetization, label="Original")
    plt.plot(temperatures, magnetization_renormalized, label="Renormalized")
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title("Flow of Fixed Points under Renormalization")
    plt.legend()
    plt.savefig('flow.png')

    # Log-Log Plot for Order Parameter near Tc
    plt.figure()
    plt.loglog(np.abs(temperatures - Tc), np.abs(magnetization), label="Original")
    plt.loglog(np.abs(temperatures - Tc), np.abs(magnetization_renormalized), label="Renormalized")
    plt.xlabel("|T - Tc|")
    plt.ylabel("|M|")
    plt.title("Log-Log Plot for Order Parameter near Tc")
    plt.legend()
    plt.savefig('order_parameter.png')

    # Spin Configurations at Tc
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(spin_configs[Tc], cmap="coolwarm")
    plt.title(f"Spin Configuration at Tc ({Tc:.2f})")
    plt.colorbar(label="Spin")
    
    plt.subplot(1, 2, 2)
    plt.imshow(spin_configs_renormalized[Tc], cmap="coolwarm")
    plt.title(f"Renormalized Spin Configuration at Tc ({Tc:.2f})")
    plt.colorbar(label="Spin")
    plt.savefig('spin_configurations.png')

    # Correlation Length vs Temperature
    plt.figure()
    plt.plot(temperatures, correlation_lengths, label="Correlation Length")
    plt.xlabel("Temperature")
    plt.ylabel("Correlation Length")
    plt.title("Correlation Length vs Temperature")
    plt.legend()
    plt.savefig('correlation_length.png')

    # Binder Cumulant vs Temperature
    plt.figure()
    plt.plot(temperatures, binder_cumulants, label="Binder Cumulant")
    plt.xlabel("Temperature")
    plt.ylabel("Binder Cumulant")
    plt.title("Binder Cumulant vs Temperature")
    plt.legend()
    plt.savefig('binder_cumulant.png')
