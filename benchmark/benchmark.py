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
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Pauli
from qiskit.circuit import ParameterVector

from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.utilities import to_qiskit_operator, get_eigenvalues, get_probability_distribution

from qiskit import QuantumCircuit

# PARAMETERS
R = 0.86
SHOTS = 1024
N_EXP = 2
N_0 = 10
SEED = None
PLOT = True
XLIM = 0.1



PauliX = Pauli("X")
PauliZ = Pauli("Z")
PauliI = Pauli("I")
PauliY = Pauli("Y")

# Molecule Type
# molecule = create_h2(R)
molecule = create_h3(R)
# molecule = create_lih(R)

# Hamiltonian
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)
# print(qiskit_hamiltonian)

num_qubits = qiskit_hamiltonian.num_qubits


# Pools
pool = QE(molecule)

# indices = [2]
indices = [18, 12, 3, 1]
# print(coefficients)
# print(indices)

parameters = ParameterVector("theta", len(indices))
coefficients = [0.11480156, -0.07351834, 0.05473366, 0.05316484]
parameters_value = [0.1148, -0.07352, 0.05473, 0.05316]

# coefficients = [0.1]*len(indices)
# parameters_value = [0.1]*len(indices)

# parameters = [ 1.148e-01 -7.352e-02  5.473e-02  5.316e-02]

ref_circuit = QuantumCircuit(num_qubits)
ref_circuit.x([0,1,2])
ref_circuit.barrier()

parameterized_circuit = pool.get_parameterized_circuit(indices, coefficients, parameters)

ansatz = ref_circuit.compose(parameterized_circuit)


# Hardware Efficient Circuit
# ansatz = EfficientSU2(num_qubits, reps=0)

# parameters_value = np.zeros(ansatz.num_parameters)

# ---


print(ansatz)

print("Number of Parameters:", ansatz.num_parameters)

# Qiskit Estimator
estimator = StatevectorEstimator()
pub = (ansatz, qiskit_hamiltonian, parameters_value)
job = estimator.run([pub])
exp_vals_estimator = job.result()[0].data.evs


# Qiskit Sampler without Shots Distribution
sampler = StatevectorSampler(seed=SEED)
commuted_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)


energy_list_exp = []
for exp in range(N_EXP):
    print(f"Experiments-{exp}")
    circuit_cliques = []
    energy_qiskit_sampler = 0

    for i, cliques in enumerate(commuted_hamiltonian):
        # print(f'Clique-{i}: {cliques.paulis}')

        circuit_clique = ansatz.copy()
        for j, pauli in enumerate(cliques[0].paulis[0]):
            # print(pauli)
            if (pauli == PauliY):
                circuit_clique.sdg(j)
                circuit_clique.h(j)
            elif (pauli == PauliX):
                circuit_clique.h(j)
        
        circuit_clique.measure_all()

        circuit_cliques.append(circuit_clique.decompose())

        job = sampler.run(pubs=[(circuit_clique, parameters_value)], shots=SHOTS)

        counts = job.result()[0].data.meas.get_counts()
        probs = get_probability_distribution(counts, SHOTS, num_qubits)

        for pauli_string in cliques:
            eigen_val = get_eigenvalues(pauli_string.to_list()[0][0])
            res = np.dot(eigen_val, probs) * pauli_string.coeffs

            energy_qiskit_sampler += res[0].real
        
    energy_list_exp.append(energy_qiskit_sampler)

# Calculate mean and standard deviation
mean_val = np.mean(energy_list_exp)
std_val = np.std(energy_list_exp)


print("\nEnergy FCI =", molecule.fci_energy)
print("Energy Estimator =", exp_vals_estimator)
print("Energy Sampler =", mean_val)

print(f"\nError Estimator = {np.abs(molecule.fci_energy-exp_vals_estimator)*627.5094} kcal/mol")
print(f"Error Samppler = {np.abs(molecule.fci_energy-mean_val)*627.5094} kcal/mol")



formatted_time = time.strftime('%d%m%y_%H%M%S', time.localtime())
os.makedirs('results', exist_ok=True)
filename = f'results/E_{molecule.description}_shots={SHOTS}_N_exp={N_EXP}_T_{formatted_time}'
with open(filename, 'w') as f:
    json.dump(energy_list_exp, f)




# Plotting
if PLOT:
    sorted_data = sorted(energy_list_exp)

    plt.figure(dpi=100)

    sns.kdeplot(sorted_data, label=f'{molecule.description}', linestyle='--')
    plt.axvline(exp_vals_estimator, linestyle='dotted', label='Statevector (Exact) Energy')

    plt.text(
        0.05, 0.95, 
        f'Mean = {mean_val:.4f}\nSTD = {std_val:.4f}\nE Exact={exp_vals_estimator:.4f}', 
        transform=plt.gca().transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    if XLIM is not None:
        plt.xlim(exp_vals_estimator-XLIM, exp_vals_estimator+XLIM)


    plt.xlabel('Calculated Energy')
    plt.title(f'{molecule.description}, {SHOTS} Shots, {N_EXP} Experiments')
    plt.legend(loc='lower right')
    plt.show()
