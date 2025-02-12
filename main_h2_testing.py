from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.hamiltonian import h_lih
from src.utilities import to_qiskit_operator
from algorithms.adapt_vqe_v8 import AdaptVQE

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator

from qiskit.quantum_info import SparsePauliOp


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

    # print("Pool:", pool)
    for i in range(len(pool.operators)):
        print(f"\nPool-{i}")
        # print("Pool:", pool.operators[i].q_operator)
        gradient = commutator(qubit_hamiltonian, pool.operators[i].q_operator)
        gradient_qiskit = to_qiskit_operator(gradient)

        gradient_list.append(gradient_qiskit)
    
    print(gradient_list)
        
