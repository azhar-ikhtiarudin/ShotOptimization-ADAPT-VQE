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

def map_eigenvalues(counts, shots):
    total = 0
    for bitstring, freq in counts.items():
        parity = sum(int(b) for b in bitstring) % 2  # Compute parity
        eigenvalue = 1 if parity == 0 else -1  # Map based on parity
        total += eigenvalue * freq
    return total / shots  # Compute expectation value

def expectation_value(counts, observable):
    total_shots = sum(counts.values())
    weighted_sum = 0

    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring]

        # Find relevant indices (where observable != 'I')
        relevant_indices = [i for i, c in enumerate(observable) if c != 'I']

        # Compute parity (sum of relevant bits mod 2)
        parity = sum(bits[i] for i in relevant_indices) % 2
        eigenvalue = 1 if parity == 0 else -1

        weighted_sum += eigenvalue * count

    return weighted_sum / total_shots


# Function to map bitstrings to eigenvalues based on parity
def counts_to_expvals(counts):
    total_shots = sum(counts.values())  # Total number of shots
    # print("Total Shots:", total_shots)
    weighted_sum = 0  # Initialize sum

    # print(counts)

    for bitstring, count in counts.items():
        # print(f'bitstring: {bitstring}, count: {count}')
        bits = [int(b) for b in bitstring]  # Convert bitstring to list of integers
        
        # Compute parity (change this if the observable has specific Y/X positions)
        parity = sum(bits) % 2  # Compute sum mod 2

        # print(f'bits: {bits} | parity: {parity}')
        
        eigenvalue = 1 if parity == 0 else -1  # Map to eigenvalue
        weighted_sum += eigenvalue * count  # Multiply by frequency

    return weighted_sum / total_shots  # Normalize by total shots


# PARAMETERS
R = 0.86
SHOTS = 1024
N_EXP = 1000
N_0 = 10
SEED = None
PLOT = False
XLIM = 0.1



PauliX = Pauli("X")
PauliZ = Pauli("Z")
PauliI = Pauli("I")
PauliY = Pauli("Y")

# Molecule Type
# molecule = create_h2(R)
molecule = create_h3(R)
# molecule = create_lih(R)

# print(molecule.fci_energy)

# breakpoint()
# Hamiltonian
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)
# print(qiskit_hamiltonian)


from qiskit.quantum_info import SparsePauliOp
qiskit_hamiltonian = SparsePauliOp.from_list([("IIII", -7.4989469), ("XXYY", -0.0029329),
                                       ("XYYX", 0.0029329), ("XZXI", 0.0129108),
                                       ("XZXZ", -0.0013743), ("XIXI", 0.0115364),
                                       ("YXXY", 0.0029329), ("YYXX", -0.0029320),
                                       ("YZYI", 0.0129108), ("YZYZ", -0.0013743),
                                       ("YIYI", 0.0115364), ("ZIII", 0.1619948),
                                       ("ZXZX", 0.0115364), ("ZYZY", 0.0115364),
                                       ("ZZII", 0.1244477), ("ZIZI", 0.0541304),
                                       ("ZIIZ", 0.0570634), ("IXZX", 0.0129108),
                                       ("IXIX", -0.0013743), ("IYZY", 0.0129107),
                                       ("IYIY", -0.0013743), ("IZII", 0.1619948),
                                       ("IZZI", 0.0570634), ("IZIZ", 0.0541304),
                                       ("IIZI", -0.0132437), ("IIZZ", 0.0847961),
                                       ("IIIZ", -0.0132436)
                                       ])

commuted_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)
# commuted_hamiltonian



num_qubits = qiskit_hamiltonian.num_qubits


# Pools
pool = QE(molecule)


# H3 Conf
indices = [18, 12, 3, 1]
parameters = ParameterVector("theta", len(indices))
coefficients = [0.11480156, -0.07351834, 0.05473366, 0.05316484]
parameters_value = [0.1148, -0.07352, 0.05473, 0.05316]


# H2 Conf
# indices = [2]
# parameters = ParameterVector("theta", len(indices))
# coefficients = [0.1]*len(indices)
# parameters_value = [0.1]*len(indices)

qc_circuit = True
qc_circuit = False

# Ansatz
if qc_circuit:
    print("Qubit-Excitation Ansatz")
    ref_circuit = QuantumCircuit(num_qubits)
    ref_circuit.x([0,1,2])
    ref_circuit.barrier()
    print(ref_circuit)
    parameterized_circuit = pool.get_parameterized_circuit(indices, coefficients, parameters)
    print(parameterized_circuit)
    ansatz = ref_circuit.compose(parameterized_circuit)
else:
    print("Hardware Efficient Ansatz")
    ansatz = EfficientSU2(num_qubits, reps=0)
    parameters_value = np.zeros(ansatz.num_parameters)

print(ansatz)
# breakpoint()

# print("Number of Parameters:", ansatz.num_parameters)


# Qiskit Estimator
estimator = StatevectorEstimator()
pub = (ansatz, qiskit_hamiltonian, parameters_value)
job = estimator.run([pub])
exp_vals_estimator = job.result()[0].data.evs


# Qiskit Sampler without Shots Distribution
sampler = StatevectorSampler(seed=SEED)
# commuted_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)

# print(commuted_hamiltonian)



cliques_measurement_list = []
E_2 = 0
expectation_value_total_sampler = 0
expectation_value_total_estimator = 0

E_cliques_list = {
    'sampler_1':[],
    'sampler_2':[],
    'estimator':[]
}

for i, cliques in enumerate(commuted_hamiltonian):
    E_cliques = {
    'sampler_1':0,
    'sampler_2':0,
    'estimator':0
    }

    print(f'Clique-{i}: {cliques.paulis}')

    circuit_clique = ansatz.copy()
    for j, pauli in enumerate(cliques[0].paulis[0]):
        # print(pauli)
        if (pauli == PauliY):
            circuit_clique.sdg(j)
            circuit_clique.h(j)
        elif (pauli == PauliX):
            circuit_clique.h(j)
    circuit_clique.measure_all()

    for exp in range(N_EXP):

        job = sampler.run(pubs=[(circuit_clique, parameters_value)], shots=SHOTS)
        counts = job.result()[0].data.meas.get_counts()

        probs = get_probability_distribution(counts, SHOTS, num_qubits)

    # print(f'\tClique-{i} | Experiment-{exp}')

    for pauli_string in cliques:
        eigen_val = get_eigenvalues(pauli_string.to_list()[0][0])
        res = np.dot(eigen_val, probs)
        exp_vals_sampler = (res*pauli_string.coeffs)[0].real


        # exp_vals_2 = counts_to_expvals(counts)
        exp_vals_2 = expectation_value(counts, pauli_string.to_list()[0][0])


        print(f'\n\n\t\tPauli string = {pauli_string.to_list()[0][0]} | res = {res:.5f} | coeffs = {pauli_string.coeffs} ')
        
        pub = (ansatz, pauli_string, parameters_value)
        job = estimator.run([pub])
        exp_vals_estimator = job.result()[0].data.evs


        print('\t\tres_1 =', res)
        print('\t\tres_2 =', exp_vals_2)

        print(f'\n\t\tSampler_1 = {exp_vals_sampler}')
        print(f'\t\tSampler_2 = {(exp_vals_2*pauli_string.coeffs)[0].real}')
        print(f'\t\tEstimator = {exp_vals_estimator}')

        E_2 += (exp_vals_2*pauli_string.coeffs)[0].real
        expectation_value_total_sampler += exp_vals_sampler
        expectation_value_total_estimator += exp_vals_estimator

        E_cliques['sampler_1'] += exp_vals_sampler
        E_cliques['sampler_2'] += (exp_vals_2*pauli_string.coeffs)[0].real
        E_cliques['estimator'] += exp_vals_estimator

        print(':', E_cliques['sampler_1'])
        print(':', E_cliques['sampler_2'])
        print(':', E_cliques['estimator'])

    E_cliques_list['sampler_1'].append(E_cliques['sampler_1'])
    E_cliques_list['sampler_2'].append(E_cliques['sampler_2'])
    E_cliques_list['estimator'].append(E_cliques['estimator'])

print(E_2)
print(expectation_value_total_sampler)
print(expectation_value_total_estimator)

print(E_cliques_list['sampler_1'])
print(E_cliques_list['sampler_2'])
print(E_cliques_list['estimator'])


print(np.sum(E_cliques_list['sampler_1']))
print(np.sum(E_cliques_list['sampler_2']))
print(np.sum(E_cliques_list['estimator']))


