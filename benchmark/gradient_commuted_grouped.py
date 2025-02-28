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
    molecule = create_h2(r)
    Hf = molecule.get_molecular_hamiltonian()
    Hq = jordan_wigner(Hf)
    Hqis = to_qiskit_operator(Hq)
    # Hqis = set(to_qiskit_operator(Hq).paulis.to_labels())

    Hqis_c = Hqis.group_commuting(qubit_wise=True)
    Hqis_c_array = []

    for i in range(len(Hqis_c)):
        Hqis_c_array.append(Hqis_c[i].paulis[0])

    # print(len(Hqis_c))
    # print(Hqis_c)
    print(len(Hqis_c_array))
    print(Hqis_c_array)

    # Operator Pool
    pool = QE(molecule)
    operator_pool = QubitOperator('')

    N_standard = 0
    N_similiar = 0

    for i in range(len(pool.operators)):
        print(f"\n\nGradient-{i}")
        Aq = pool.operators[i]._q_operator
        grad_obs = commutator(Hq, Aq)
        grad_obs_qis_c = to_qiskit_operator(grad_obs).group_commuting(qubit_wise=True)
        print("Full Gradient Observable:", grad_obs_qis_c)
        grad_obs_qis_c_array = []

        for j in range(len(grad_obs_qis_c)):
            grad_obs_qis_c_array.append(grad_obs_qis_c[j].paulis[0])
        
        print("Grad 2:", grad_obs_qis_c_array)

        for k in range(len(Hqis_c_array)):
            print(Hqis_c_array[k])
            


        




        # for pauli in grad_obs_qis:
            # print(pauli)
            # print(type(pauli))

            
            # N_standard += 1
            # if pauli in Hqis:
            #     print(f"{pauli} recycled")
            #     N_similiar += 1
    
    # print(N_standard)
    # print(N_standard-N_similiar)
    # print(f'{(N_standard-N_similiar)/N_standard*100} %')