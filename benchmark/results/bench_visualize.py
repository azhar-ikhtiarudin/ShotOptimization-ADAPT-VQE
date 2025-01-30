import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_filename(filename):
    # Extract relevant data using regex
    match = re.match(r"E_(\w+)_shots=(\d+)_N_exp=(\d+)_T_(\d+)", filename)
    if match:
        mol_name = match.group(1)
        mol_shots = match.group(2)
        mol_exp = match.group(3)
        return mol_name, mol_shots, mol_exp
    else:
        raise ValueError("Filename format is incorrect.")

def plot_probability_density(filename, xlim=None):    

    mol_name, mol_shots, mol_exp = parse_filename(filename)

    with open(filename, 'r') as f:
        data = json.load(f)

    mean_val = np.mean(data)
    std_val = np.std(data)

    if mol_name == 'H2':
        state_vector_energy = 0.713176834123989
    elif mol_name == 'H3':
        state_vector_energy = -1.5658
    elif mol_name == 'LiH':
        state_vector_energy = 2.13953050237196

    # for i in range(len(data)):
    sorted_data = sorted(data)

    # Plot Probability Density
    sns.kdeplot(sorted_data, label=f'H$_2$', linestyle='--')
    
    # Add Text
    plt.text(
        0.05, 0.95, 
        f'Mean = {mean_val:.4f}\nSTD = {std_val:.4f}\nE Exact={state_vector_energy:.4f}\nxlim={xlim}', 
        transform=plt.gca().transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    # Add Statevector Energy
    plt.axvline(state_vector_energy, linestyle='dotted', label='Statevector (Exact) Simulation', color='black')

    # Conf Plot
    if xlim is not None:
        plt.xlim(state_vector_energy - xlim, state_vector_energy + xlim)
    plt.xlabel('Calculated Energy')
    plt.title(f'{mol_name}, {mol_shots} Shots, {mol_exp} Experiments')
    plt.legend(loc='lower right')
    plt.show()


filename = 'E_H3_shots=1024_N_exp=100000_T_290125_021506'

plot_probability_density(filename, xlim=None)