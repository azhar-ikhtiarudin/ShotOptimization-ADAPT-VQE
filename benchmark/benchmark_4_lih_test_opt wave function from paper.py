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
# qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

from openfermion.ops import QubitOperator

qubit_hamiltonian =  (
            -7.4989469 * QubitOperator('') +
            -0.0029329 * QubitOperator('Y0 Y1 X2 X3') +
            0.0029329 * QubitOperator('Y0 X1 X2 Y3') +
            0.0129108 * QubitOperator('X0 Z1 X2') +
            -0.0013743 * QubitOperator('X0 Z1 X2 Z3') +
            0.0115364 * QubitOperator('X0 X2') +
            0.0029329 * QubitOperator('X0 X1 Y2 Y3') +
            -0.0029320 * QubitOperator('X0 Y1 Y2 X3') +
            0.0129108 * QubitOperator('Y0 Z1 Y2') +
            -0.0013743 * QubitOperator('Y0 Z1 Y2 Z3') +
            0.0115364 * QubitOperator('Y0 Y2') +
            0.1619948 * QubitOperator('Z0') +
            0.0115364 * QubitOperator('X0 Z1 X2 Z3') +
            0.0115364 * QubitOperator('Y0 Z1 Y2 Z3') +
            0.1244477 * QubitOperator('Z0 Z1') +
            0.0541304 * QubitOperator('Z0 Z2') +
            0.0570634 * QubitOperator('Z0 Z3') +
            0.0129108 * QubitOperator('X1 Z2 X3') +
            -0.0013743 * QubitOperator('X1 X2') +
            0.0129107 * QubitOperator('Y1 Z2 Y3') +
            -0.0013743 * QubitOperator('Y1 Y2') +
            0.1619948 * QubitOperator('Z1') +
            0.0570634 * QubitOperator('Z1 Z2') +
            0.0541304 * QubitOperator('Z1 Z3') +
            -0.0132437 * QubitOperator('Z2') +
            0.0847961 * QubitOperator('Z2 Z3') +
            -0.0132436 * QubitOperator('Z3')
        )
qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)
print(qiskit_hamiltonian)

num_qubits = qiskit_hamiltonian.num_qubits
print(num_qubits)


# Pools

parameters = ParameterVector("theta", 8)
parameters_value = [-0.032, 0.546, -1.587, 1.567, 0.015, -0.534, -1.570, -1.556]

ansatz = QuantumCircuit(4)

ansatz.x(0)
ansatz.x(1)

ansatz.ry(parameters[0], 0)
ansatz.ry(parameters[1], 1)
ansatz.ry(parameters[2], 2)
ansatz.ry(parameters[3], 3)

ansatz.cx(0,1)
ansatz.cx(1,2)
ansatz.cx(2,3)

ansatz.ry(parameters[4], 0)
ansatz.ry(parameters[5], 1)
ansatz.ry(parameters[6], 2)
ansatz.ry(parameters[7], 3)

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
commuted_hamiltonian = qiskit_hamiltonian.group_commuting(qubit_wise=True)

# print(commuted_hamiltonian)



cliques_measurement_list = []

expectation_value_total_sampler = 0
expectation_value_total_estimator = 0

E_cliques_list = {
    'sampler':[],
    'estimator':[]
}

for i, cliques in enumerate(commuted_hamiltonian):
    E_cliques = {
    'sampler':0,
    'estimator':0
    }

    print(f'\n\n. . . ====== Clique-{i} ====== . . .')

    circuit_clique = ansatz.copy()
    for j, pauli in enumerate(cliques[0].paulis[0]):
        if (pauli == PauliY):
            circuit_clique.sdg(j)
            circuit_clique.h(j)
        elif (pauli == PauliX):
            circuit_clique.h(j)
    circuit_clique.measure_all()

    for k, pauli_string in enumerate(cliques):
        exp_val_sampler_list = []

        for exp in range(N_EXP):

            job = sampler.run(pubs=[(circuit_clique, parameters_value)], shots=SHOTS)
            counts = job.result()[0].data.meas.get_counts()
            probs = get_probability_distribution(counts, SHOTS, num_qubits)

            eigen_val = get_eigenvalues(pauli_string.to_list()[0][0])
            res = np.dot(eigen_val, probs)
            exp_vals_sampler = (res*pauli_string.coeffs)[0].real

            print(f'\nClique-{i} | Pauli: {pauli_string.to_list()[0][0]} | Experiment-{exp}')
            
            pub = (ansatz, pauli_string, parameters_value)
            job = estimator.run([pub])
            exp_vals_estimator = job.result()[0].data.evs

            print(f'\tres = {res:.5f} | sampler = {exp_vals_sampler} | estimator = {exp_vals_estimator} ')

            exp_val_sampler_list.append(exp_vals_sampler)
        
        print(f'\nAverage Value Clique-{i} | Pauli: {pauli_string.to_list()[0][0]}')
        print(f'\tres = {res:.5f} | sampler = {np.mean(exp_val_sampler_list)} | estimator = {exp_vals_estimator} ')


        expectation_value_total_sampler += np.mean(exp_vals_sampler)
        expectation_value_total_estimator += exp_vals_estimator


    # print('\t\tres_1 =', res)
    # print(f'\n\t\tSampler_1 = {exp_vals_sampler}')
    # print(f'\t\tEstimator = {exp_vals_estimator}')

    # expectation_value_total_sampler += exp_vals_sampler
    # expectation_value_total_estimator += exp_vals_estimator

    # E_cliques['sampler'] += exp_vals_sampler
    # E_cliques['estimator'] += exp_vals_estimator

    # print(':', E_cliques['sampler'])
    # print(':', E_cliques['estimator'])

    # E_cliques_list['sampler'].append(E_cliques['sampler'])
    # E_cliques_list['estimator'].append(E_cliques['estimator'])

print('\nExpectation Value Total Sampler', expectation_value_total_sampler)
print('Expectation Value Total Estimator', expectation_value_total_estimator)

# print(E_cliques_list['sampler'])
# print(E_cliques_list['estimator'])

# print(np.sum(E_cliques_list['sampler']))
# print(np.sum(E_cliques_list['estimator']))


