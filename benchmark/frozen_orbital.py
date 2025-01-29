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
from openfermion.transforms import jordan_wigner, get_fermion_operator, freeze_orbitals
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
N_EXP = 50
N_0 = 10
SEED = None
PLOT = True
XLIM = 0.1

PauliX = Pauli("X")
PauliZ = Pauli("Z")
PauliI = Pauli("I")
PauliY = Pauli("Y")

# Molecule Type
molecule = create_h2(R)
# molecule = create_h3(R)
molecule = create_lih(R)

# Hamiltonian
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()

print(molecule.n_orbitals)

qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

qiskit_hamiltonian = to_qiskit_operator(qubit_hamiltonian)

print(qiskit_hamiltonian.num_qubits)


# frozen_orbitals = [0,1]
# freezed_hamiltonian = get_fermion_operator(fermionic_hamiltonian)
# # print("Get Fermion Operator", hamiltonian)
# freezed_hamiltonian = freeze_orbitals(freezed_hamiltonian, frozen_orbitals)
# # print("Freeze Orbitals", hamiltonian)
# qubit_hamiltonian_freezed = jordan_wigner(freezed_hamiltonian)
# qiskit_hamiltonian_freezed = to_qiskit_operator(qubit_hamiltonian_freezed)

# print(qiskit_hamiltonian_freezed.num_qubits)

