import sys
import os
sys.path.append('/home/azhar04/project/1. dev/quantum-dev/ShotOptimized-ADAPT-VQE/')

from src.pools import SD, GSD, GSD1, SingletGSD, SpinCompGSD, PauliPool,  NoZPauliPool1, NoZPauliPool, QE, QE1, QE_All, CEO, OVP_CEO, DVG_CEO, DVE_CEO, MVP_CEO
from src.molecules import create_h2, create_h3, create_h4, create_lih
from src.hamiltonian import h_lih
from src.utilities import to_qiskit_operator

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator
from openfermion.ops import QubitOperator

# from algorithms.adapt_vqe_v9_grad import AdaptVQE
from algorithms.adapt_vqe_v10_meas_recycle import AdaptVQE


def calculate_recycle(Hq, Gq):
    
    pass


if __name__ == '__main__':    
    
    # Molecular Hamiltonian
    r = 0.742
    molecule = create_h3(r)
    Hf = molecule.get_molecular_hamiltonian()
    Hq = jordan_wigner(Hf)
    Hqis = set(to_qiskit_operator(Hq).paulis.to_labels())

    print(Hqis)

    # Operator Pool
    pool = QE(molecule)
    operator_pool = QubitOperator('')

    N_standard = len(Hqis)
    N_reduced = len(Hqis)
    N_similiar = 0
    print("N_standard:",N_standard)
    print("N_reduced:",N_reduced)

    for i in range(len(pool.operators)):
        Aq = pool.operators[i]._q_operator
        grad_obs = commutator(Hq, Aq)
        grad_obs_qis = to_qiskit_operator(grad_obs).paulis.to_labels()
        print(f"\n\tGradient-{i}: {grad_obs_qis} | len = {len(grad_obs_qis)}")

        for pauli in grad_obs_qis:
            N_standard += 1
            if pauli in Hqis:
                print(f"\t\t{pauli} recycled")
                N_similiar += 1
    
    print(N_standard)
    print(N_standard-N_similiar)
    print(f'{(N_standard-N_similiar)/N_standard*100} %')