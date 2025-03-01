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
    print(Hqis_c)
    print(len(Hqis_c_array))
    print(Hqis_c_array)

    # Operator Pool
    pool = QE(molecule)
    operator_pool = QubitOperator('')


    N_standard = len(Hqis)
    N_reduced = len(Hqis_c_array)


    print("\tN Standard:", N_standard)
    print("\tN Reduced:", N_reduced)


    # for i in range(1):
    for i in range(len(pool.operators)):
        print(f"\n\n# Gradient-{i} 📈 ")
        Aq = pool.operators[i]._q_operator
        grad_obs = commutator(Hq, Aq)
        grad_obs_qis = to_qiskit_operator(grad_obs)
        print(" > Full Gradient Observable:", grad_obs_qis.paulis)
        # N_standard += len(grad_obs_qis)

        # Group Commuting
        grad_obs_qis_c = grad_obs_qis.group_commuting(qubit_wise=True)
        # print(" > Commuting Gradient Observable:", grad_obs_qis_c)

        for g in grad_obs_qis_c:
            print("\n\tLoop through commuting gradient:", g.paulis)
            
            # Check if h is commute with one of avaiable measurements
            for h in Hqis_c_array:
                is_commute = False
                print("\t\t> Compare with",h)
                print("\t\t  g", g[0].paulis)
                # print("\t\tType of g", type(g[0]))
                # print("\t\tType of h", type(h))
                is_commute = is_commute | g[0].paulis.commutes(h)
                print(f'\t\t  Is Commute? {is_commute}')
                if is_commute:
                    print("\t\t\tCommuted ✅")
                    break

            if is_commute:
                N_reduced += 0
                N_standard += len(g.paulis)
                print("\tN Standard:", N_standard)
                print("\tN Reduced:", N_reduced)
            else:
                N_reduced += len(g.paulis)
                N_standard += len(g.paulis)
                
            print("\tN Standard:", N_standard)
            print("\tN Reduced:", N_reduced)

        print("\tFinal N Standard:", N_standard)
        print("\tFinal N Reduced:", N_reduced)

