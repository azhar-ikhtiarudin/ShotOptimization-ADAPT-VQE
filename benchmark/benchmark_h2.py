import sys
import os
sys.path.append('/home/azhar04/project/1. dev/quantum-dev/ShotOptimized-ADAPT-VQE/')
# sys.path.append('/home/alfarialstudio/ShotOptimization-ADAPT-VQE/')

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpyscf import run_pyscf

from qiskit import QuantumCircuit
# from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Pauli
from qiskit.circuit import ParameterVector

from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.utilities import to_qiskit_operator, get_eigenvalues, get_probability_distribution

from utils import get_uniform_shots_dist, get_variance_shots_dist, calculate_exp_value_sampler
from utils import h_lih

from qiskit import QuantumCircuit

PauliX = Pauli("X")
PauliZ = Pauli("Z")
PauliI = Pauli("I")
PauliY = Pauli("Y")


# PARAMETERS
R = 0.742
SHOTS = 1024
N_EXP = 10
N_0 = 10 # CHANGE THIS to 10, 50, 100
SEED = None

R = 0.742
molecule = create_h2(R)

fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

qubit_hamiltonian = h_lih

qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)
molecule_description = 'H2'
num_qubits = qiskit_hamiltonian.num_qubits
print(num_qubits)



circuit = efficient_su2(num_qubits, reps=0)
print(circuit)
num_params = circuit.num_parameters
params = np.zeros(num_params)


# === Qiskit Estimator ===
estimator = Estimator()
pub = (circuit, qiskit_hamiltonian, params)
job = estimator.run(pubs=[pub])
energy_statevector = job.result()[0].data.evs




# === Qiskit Sampler with Shots Distribution ===

sampler = Sampler(seed=SEED)
commuted_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)

shots_budget = SHOTS*len(commuted_hamiltonian)
print("Total Shots Budget:", shots_budget)


# breakpoint()

energy_uniform_list = []
energy_vpsr_list = []
energy_vmsa_list = []

for _ in range(N_EXP):
    uniform_shots_dist = get_uniform_shots_dist(shots_budget, len(commuted_hamiltonian))
    vmsa_shots_dist = get_variance_shots_dist(commuted_hamiltonian, shots_budget, 
                                            'vmsa',N_0, circuit, params, 
                                            sampler, num_qubits)
    vpsr_shots_dist = get_variance_shots_dist(commuted_hamiltonian, shots_budget, 
                                            'vpsr',N_0, circuit, params, 
                                            sampler, num_qubits)
    
    energy_uniform = calculate_exp_value_sampler(commuted_hamiltonian, params, circuit, uniform_shots_dist, sampler, num_qubits)
    energy_vpsr = calculate_exp_value_sampler(commuted_hamiltonian, params, circuit, vpsr_shots_dist, sampler, num_qubits)
    energy_vmsa = calculate_exp_value_sampler(commuted_hamiltonian, params, circuit, vmsa_shots_dist, sampler, num_qubits)
    energy_uniform_list.append(energy_uniform)
    energy_vpsr_list.append(energy_vpsr)
    energy_vmsa_list.append(energy_vmsa)


energy_uniform = np.mean(energy_uniform_list)
energy_vpsr = np.mean(energy_vpsr_list)
energy_vmsa = np.mean(energy_vmsa_list)

std_uniform = np.std(energy_uniform_list)
std_vpsr = np.std(energy_vpsr_list)
std_vmsa = np.std(energy_vmsa_list)

print("\nEnergy Statevector", energy_statevector)


print("Energy Uniform Mean", energy_uniform)
print("Energy VPSR Mean", energy_vpsr)
print("Energy VMSA Mean", energy_vmsa)

print("Energy Uniform STD", std_uniform)
print("Energy VPSR STD", std_vpsr)
print("Energy VMSA STD", std_vmsa)


sorted_uniform = sorted(energy_uniform_list)
sorted_vpsr = sorted(energy_vpsr_list)
sorted_vmsa = sorted(energy_vmsa_list)

print(type(energy_statevector))
print(energy_statevector)
# print(type(sorted_uniform))
# print(type(sorted_vpsr))

data = {
    'statevector': energy_statevector.tolist(),
    'uniform': sorted_uniform,
    'vpsr': sorted_vpsr,
    'vmsa':sorted_vmsa
}


formatted_time = time.strftime('%d%m%y_%H%M%S', time.localtime())

filename = f'E_{molecule_description}_shots={SHOTS}_N_0={N_0}_N_exp={N_EXP}_T_{formatted_time}.json'
with open(filename, 'w') as f:
    json.dump(data, f)
print(f"Data saved to {filename}")


# Quick Plot for Preview

plt.figure(dpi=100)

sns.kdeplot(sorted_vmsa, label=f'{molecule_description} VMSA', linestyle='--')
sns.kdeplot(sorted_vpsr, label=f'{molecule_description} VPSR', linestyle='--')
sns.kdeplot(sorted_uniform, label=f'{molecule_description} Uniform', linestyle='--')
# sns.kdeplot(sorted_data, label=f'{molecule_description}', linestyle='--')

plt.axvline(energy_statevector, linestyle='dotted', label='Statevector (Exact) Energy')

# plt.text(
#     0.05, 0.95, 
#     f'Mean = {mean_val:.4f}\nSTD = {std_val:.4f}\nE Exact={exp_vals_estimator:.4f}', 
#     transform=plt.gca().transAxes, 
#     fontsize=10, 
#     verticalalignment='top',
#     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
# )


plt.xlabel('Calculated Energy')
plt.title(f'{molecule_description}, {SHOTS} Shots, {N_EXP} Experiments')
plt.legend(loc='lower right')
plt.show()
