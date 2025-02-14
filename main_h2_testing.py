from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.hamiltonian import h_lih
from src.utilities import to_qiskit_operator
from algorithms.adapt_vqe_v8 import AdaptVQE

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator

from qiskit.quantum_info import SparsePauliOp, PauliList
import numpy as np


if __name__ == '__main__':    
    r = 0.742
    molecule = create_h2(r)
    
    # Hamiltonian
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    # print(f"Qubit Hamiltonian:", qubit_hamiltonian)

    # Pool
    pool = QE(molecule=molecule)
    gradient_list = []
    # pauli_list = []
    pauli_list = PauliList(["IIII"])
    coeff_list = np.array([])

    # print("Pool:", pool)
    for i in range(len(pool.operators)):
        print(f"\nPool-{i}")
        # print("Pool:", pool.operators[i].q_operator)
        gradient = commutator(qubit_hamiltonian, pool.operators[i].q_operator)
        gradient_qiskit = to_qiskit_operator(gradient)

        # print("Pauli Strings:", gradient_qiskit._pauli_list)
        # print("Coefficients:", gradient_qiskit.coeffs)
        # print("Type - Pauli Strings:", type(gradient_qiskit._pauli_list))
        print("Type Coefficients:", type(gradient_qiskit.coeffs))

        gradient_list.append(gradient_qiskit)
        # pauli_list.append(gradient_qiskit._pauli_list)
        pauli_list = pauli_list.insert(len(pauli_list), gradient_qiskit._pauli_list)
        # coeff_list.append(gradient_qiskit.coeffs.tolist())
        coeff_list = np.concatenate((coeff_list, gradient_qiskit.coeffs))

        print("Pauli List:", pauli_list)

    
    # print("\nList of Gradient:", gradient_list)
    pauli_list = pauli_list.delete(0)
    print("\nPauli list:", pauli_list)
    print("\nCoeff list:", coeff_list)

    gradient_obs_list = SparsePauliOp(pauli_list, coeff_list)
    # print("\nGradient Obs List:", gradient_obs_list)
    commuted_gradient_obs_list = gradient_obs_list.group_commuting(qubit_wise=True)
    # print(commuted_gradient_obs_list)

    print(len(commuted_gradient_obs_list))
    print(len(gradient_obs_list))

    print(commuted_gradient_obs_list)

    # setiap observable gradien terdiri dari beberapa pauli string, 
    # mungkin masing2 itu bisa direpresentasikan dalam indeks
    # G1:0-3 | G2:4-10 | GN:...
        
