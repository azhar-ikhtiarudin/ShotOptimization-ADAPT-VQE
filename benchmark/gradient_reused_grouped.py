import sys
import os
sys.path.append('/home/azhar04/project/1. dev/quantum-dev/ShotOptimized-ADAPT-VQE/')
sys.path.append('/home/alfarialstudio/ShotOptimization-ADAPT-VQE/')
from src.pools import SD, GSD, GSD1, SingletGSD, SpinCompGSD, PauliPool,  NoZPauliPool1, NoZPauliPool, QE, QE1, QE_All, CEO, OVP_CEO, DVG_CEO, DVE_CEO, MVP_CEO
from src.molecules import create_h2, create_h3, create_h4, create_lih
from src.hamiltonian import h_lih
from src.utilities import to_qiskit_operator

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator
from openfermion.ops import QubitOperator

# from algorithms.adapt_vqe_v9_grad import AdaptVQE
from algorithms.adapt_vqe_v10_meas_recycle import AdaptVQE


def is_qubitwise_commuting(pauli1: str, pauli2: str) -> bool:
    """
    Check if two Pauli strings are qubit-wise commuting.
    :param pauli1: First Pauli string (e.g., "IXYZ").
    :param pauli2: Second Pauli string (e.g., "XZIZ").
    :return: True if they are qubit-wise commuting, False otherwise.
    """
    if len(pauli1) != len(pauli2):
        raise ValueError("Pauli strings must have the same length.")
    
    commuting_pairs = {('I', 'Z'), ('I', 'I'),
                    ('X', 'X'), ('Y', 'Y'), ('Z', 'Z')}
    
    for p1, p2 in zip(pauli1, pauli2):
        if (p1, p2) not in commuting_pairs and (p2, p1) not in commuting_pairs:
            return False
    
    return True

# Example usage:
pauli_a = "IXYZ"
pauli_b = "XZIZ"
print(is_qubitwise_commuting(pauli_a, pauli_b))  # Output: False


def calculate_recycle(Hq, Gq):
    
    pass


if __name__ == '__main__':    
    
    # Molecular Hamiltonian
    r = 1.542
    molecule = create_h3(r)
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
    pool = DVG_CEO(molecule) # SingletGSD, SpinCompGSD, PauliPool,  NoZPauliPool1, NoZPauliPool, QE, QE1, QE_All, CEO, OVP_CEO, DVG_CEO, DVE_CEO, MVP_CEO
    operator_pool = QubitOperator('')

    N_standard_H = len(Hqis)
    N_grouped_H = len(Hqis_c_array)
    N_reduced_H = len(Hqis_c_array)
    
    N_standard_G = 0
    N_grouped_G = 0
    N_reduced_G = 0

    print("\tN Standard H:", N_standard_H)
    print("\tN Grouped H:", N_grouped_H)
    print("\tN Reduced H:", N_reduced_H)
    print("\tN Standard G:", N_standard_G)
    print("\tN Grouped G:", N_grouped_G)
    print("\tN Reduced G:", N_reduced_G)


    # for i in range(1):
    grad_group_cost = 0
    print("Len:", len(pool.operators))
    for i in range(len(pool.operators)):
        print(f"\n\n# Gradient-{i} 📈 ")
        
        Aq = pool.operators[i]._q_operator
        # Aq = jordan_wigner(pool.operators[i]._f_operator)

        grad_obs = commutator(Hq, Aq)
        grad_obs_qis = to_qiskit_operator(grad_obs)
        num_qubits = grad_obs_qis.num_qubits

        print(" > Full Gradient Observable:", grad_obs_qis.paulis)
        print(" > Full Gradient Observable Len:", len(grad_obs_qis.paulis))
        # breakpoint()
        # N_standard += len(grad_obs_qis)

        # Group Commuting
        grad_obs_qis_c = grad_obs_qis.group_commuting(qubit_wise=True)
        grad_group_cost += len(grad_obs_qis_c)
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
                # print("Num of qubits:", )
                # is_commute = is_commute | g[0].paulis.commutes(h, qargs=range(num_qubits))
                # print(type(g[0].paulis[0]))
                # print(type(h))
                # print(str(g[0][0].paulis))
                # print(str(h))
                # breakpoint()
                is_commute = is_commute | is_qubitwise_commuting(str(g[0].paulis[0]), str(h))
                print(f'\t\t  Is Commute? {is_commute}')
                if is_commute:
                    print("\t\t\tCommuted ✅")
                    break

            if is_commute:
                N_standard_G += len(g.paulis)
                N_grouped_G += 1
                N_reduced_G += 0
                print("\tN Standard:", N_standard_G)
                print("\tN Grouped:", N_grouped_G)
                print("\tN Reduced:", N_reduced_G)
            else:
                N_standard_G += len(g.paulis)
                N_grouped_G += 1
                N_reduced_G += 1
                # print("\tN Standard:", N_standard)
                # print("\tN Reduced:", N_reduced)

                print("\tN Standard:", N_standard_G)
                print("\tN Standard Grouped:", N_grouped_G)
                print("\tN Grouped Reused:", N_reduced_G)

        # print(grad_group_cost)
        print("\n\tFinal Measurement Cost")
        print("\tFinal N Standard H:", N_standard_H)
        print("\tFinal N Grouped H:", N_grouped_H)
        print("\tFinal N Reduced H:", N_reduced_H)
        print("\tFinal N Standard G:", N_standard_G)

        print("\tFinal N Grouped G:", N_grouped_G)
        print("\tFinal N Reduced G:", N_reduced_G)

        print(f"Standard: {N_standard_H+N_standard_G}")
        print(f"Grouped Commuting: {N_grouped_H+N_grouped_G}")
        print(f"Measurement Reusing: {N_grouped_H+N_reduced_G}")
        
        print(f"Standard: {N_standard_H+N_standard_G}")
        print(f"Grouped Commuting: {(N_grouped_H+N_grouped_G)/(N_standard_H+N_standard_G)*100}")
        print(f"Measurement Reusing: {(N_grouped_H+N_reduced_G)/(N_standard_H+N_standard_G)*100}")
