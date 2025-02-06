import numpy as np
from openfermion import (
    count_qubits,
    FermionOperator,
    QubitOperator,
    get_fermion_operator,
    InteractionOperator,
    jordan_wigner,
)
from qiskit.quantum_info import Pauli, SparsePauliOp

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpyscf import run_pyscf
from qiskit import QuantumCircuit

I = SparsePauliOp("I")
X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")

PauliX = Pauli("X")
PauliZ = Pauli("Z")
PauliI = Pauli("I")
PauliY = Pauli("Y")

from openfermion.ops import QubitOperator



h_lih =  (
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




def calculate_exp_value_sampler(commuted_hamiltonian, params, ansatz, 
                                shots, sampler, num_qubits):

    ansatz_cliques = []
    energy_qiskit_sampler = 0.0
    
    for i, cliques in enumerate(commuted_hamiltonian):

        ansatz_clique = ansatz.copy()
        for j, pauli in enumerate(cliques[0].paulis[0]):
            if (pauli == PauliY):
                ansatz_clique.sdg(j)
                ansatz_clique.h(j)
            elif (pauli == PauliX):
                ansatz_clique.h(j)

        ansatz_clique.measure_all()
        # ansatz_clique.measure(range(self.n), meas)

        ansatz_cliques.append(ansatz_clique)

        job = sampler.run(pubs=[(ansatz_clique, params)], shots = shots[i])

        counts = job.result()[0].data.meas.get_counts()
        # print("Counts:", counts)

        probs = get_probability_distribution(counts, shots[i], num_qubits)

        for pauli_string in cliques:
            eigen_value = get_eigenvalues(pauli_string.to_list()[0][0])
            # print("Eigen Value:", eigen_value)
            # print("Probs:", probs)
            
            res = np.dot(eigen_value, probs) * pauli_string.coeffs
            
            energy_qiskit_sampler += res[0].real
        
    return energy_qiskit_sampler



def get_uniform_shots_dist(N, l):
    shots = [ N // l ] * l
    for i in range(N % l): shots[i] += 1
    return shots

    
def get_variance_shots_dist(commuted_hamiltonian, shots_budget, type, N_0, circuit, params, sampler, num_qubits):

    circuit_cliques = []

    std_cliques = []
    for i, cliques in enumerate(commuted_hamiltonian):
        # print(cliques)
        circuit_clique = circuit.copy()
        for j, pauli in enumerate(cliques[0].paulis[0]):
            if (pauli == PauliY):
                circuit_clique.sdg(j)
                circuit_clique.h(j)
            elif (pauli == PauliX):
                circuit_clique.h(j)
        

        circuit_clique.measure_all()
        circuit_cliques.append(circuit_clique)

        job = sampler.run(pubs=[(circuit_clique, params)], shots = N_0)

        bitstrings = job.result()[0].data.meas.get_bitstrings()

        results_array = convert_bitstrings_to_arrays(bitstrings, num_qubits)

        results_one_clique = []
        for m, count_res in enumerate(results_array):
            # print(f"\nResults of shot-{m+1}")
            exp_pauli_clique = []
            for pauli_string in cliques:
                eigen_value = get_eigenvalues(pauli_string.to_list()[0][0])
                res = np.dot(eigen_value, count_res) * pauli_string.coeffs
                exp_pauli_clique.append(res[0].real)
            results_one_clique.append(np.sum(exp_pauli_clique))
        
        print(f"\nResults of Clique-{i}", results_one_clique)
        print(f"\nSTD of Clique-{i} = {np.std(results_one_clique):.4f}")
        std_cliques.append(np.std(results_one_clique))

    if sum(std_cliques) == 0:
        ratio_for_theta = [1/3 for _ in std_cliques]
    else:
        ratio_for_theta = [ v/sum(std_cliques) for v in std_cliques]
    
    print("\t\tRatio for Theta", ratio_for_theta)

    # Shots Assignment Equations
    if type == 'vmsa':
        print("N_0", N_0)
        print("std cliques", len(std_cliques))
        new_shots_budget = (shots_budget - N_0*len(std_cliques))
    elif type == 'vpsr':
        print("std cliques", len(std_cliques))
        new_shots_budget = (shots_budget - N_0*len(std_cliques))*sum(ratio_for_theta)**2/len(std_cliques)/sum([v**2 for v in ratio_for_theta])
    
    print("\t\tNew Shots budget:",new_shots_budget)
    new_shots = [max(1, round(new_shots_budget * ratio_for_theta[i])) for i in range(len(std_cliques))]

    return new_shots


    
def convert_bitstrings_to_arrays(bitstrings, N):
    all_possible_outcomes = [''.join(format(i, '0' + str(N) + 'b')) for i in range(2**N)]
    outcome_to_index = {outcome: idx for idx, outcome in enumerate(all_possible_outcomes)}
    # Convert each bitstring to a result array
    results = []
    for bitstring in bitstrings:
        result_array = [0] * (2**N)
        if bitstring in outcome_to_index:
            result_array[outcome_to_index[bitstring]] = 1
        results.append(result_array)

    return results

    
def create_h2(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h2 (PyscfMolecularData): the linear H2 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h2 = MolecularData(geometry, basis, multiplicity, charge, description='H2')
    h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)

    return h2

def create_h3(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h3 (PyscfMolecularData): the linear H3 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]], ['H', [0, 0, 2 * r]]]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    h3 = MolecularData(geometry, basis, multiplicity, charge, description='H3')
    h3 = run_pyscf(h3, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?

    return h3


def create_lih(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        lih (PyscfMolecularData): the LiH molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['Li', [0, 0, 0]], ['H', [0, 0, r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    li_h = MolecularData(geometry, basis, multiplicity, charge, description='LiH')
    li_h = run_pyscf(li_h, run_fci=True, run_ccsd=True)

    return li_h




def get_probability_distribution(counts, NUM_SHOTS, N):
    # Generate all possible N-qubit measurement outcomes
    all_possible_outcomes = [''.join(format(i, '0' + str(N) + 'b')) for i in range(2**N)]
    # Ensure all possible outcomes are in counts
    for k in all_possible_outcomes:
        if k not in counts.keys():
            counts[k] = 0
    
    # Sort counts by outcome
    sorted_counts = sorted(counts.items())
    # print("Sorted Counts", sorted_counts)
    
    # Calculate the probability distribution
    output_distr = [v[1] / NUM_SHOTS for v in sorted_counts]
    
    return output_distr


def get_eigenvalues(pauli_strings):
    # Define Pauli matrices
    eigen_I = np.array([1, 1])
    eigen_X = np.array([1, -1])
    eigen_Y = np.array([1, -1])
    eigen_Z = np.array([1, -1])

    # Map string characters to Pauli matrices
    pauli_dict = {'I': eigen_I, 'X': eigen_X, 'Y': eigen_Y, 'Z': eigen_Z}

    eigen_vals = 1
    
    for pauli in pauli_strings:
        eigen_vals = np.kron(eigen_vals, pauli_dict[pauli])
    
    return eigen_vals





def to_qiskit_pauli(letter):
    """
    Transforms a letter representing a Pauli operator to the corresponding
    Qiskit observable.

    Arguments:
        letter (str): the letter representing the Pauli operator
    Returns:
        qiskit_Pauli (PauliOp): the corresponding operator in Qiskit
    """
    if letter == "X":
        qiskit_pauli = X
    elif letter == "Y":
        qiskit_pauli = Y
    elif letter == "Z":
        qiskit_pauli = Z
    else:
        raise ValueError(
            "Letter isn't recognized as a Pauli operator" " (must be X, Y or Z)."
        )

    return qiskit_pauli


def to_qiskit_term(of_term, n, switch_endianness):
    """
    Transforms an Openfermion term into a Qiskit Operator.
    Only works for individual Pauli strings. For generic operators, see to_qiskit_operator.

    Arguments:
        of_term (QubitOperator): a Pauli string multiplied by a coefficient, given as an Openfermion operator
        n (int): the size of the qubit register
        switch_endianness (bool): whether to revert the endianness
    Returns:
        qiskit_op (PauliSumOp): the original operator, represented in Qiskit
    """

    pauli_strings = list(of_term.terms.keys())

    if len(pauli_strings) > 1:
        raise ValueError(
            "Input must consist of a single Pauli string."
            " Use to_qiskit_operator for other operators."
        )
    pauli_string = pauli_strings[0]

    coefficient = of_term.terms[pauli_string]
    

    

    qiskit_op = None

    previous_index = -1
    # print("pauli string", pauli_string)
    for qubit_index, pauli in pauli_string:
        # print("AA")
        id_count = qubit_index - previous_index - 1
    
        if switch_endianness:
            new_ops = to_qiskit_pauli(pauli)
            for _ in range(id_count):
                new_ops = new_ops ^ I
            if qiskit_op is None:
                qiskit_op = new_ops
            else:
                qiskit_op = new_ops ^ qiskit_op
        else:
            new_ops = (I ^ id_count) ^ to_qiskit_pauli(pauli)
            qiskit_op = qiskit_op ^ new_ops
        # print("--qiskit_op-1", qiskit_op)
        previous_index = qubit_index

    id_count = (n - previous_index - 1)
    # print("BB", switch_endianness)
    if switch_endianness:
        # print(id_count)
        for _ in range(id_count):
            # print("--I", I)
            # print("--qiskit_op-2", qiskit_op)
            qiskit_op = I ^ qiskit_op
    else:
        for _ in range(id_count):
            qiskit_op = qiskit_op ^ I
    
    # print("coefficient", coefficient)
    # print("qiskit_op", qiskit_op)

    qiskit_op = coefficient * qiskit_op

    return qiskit_op


def to_qiskit_operator(of_operator, n=None, little_endian=True):
    """
    Transforms an Openfermion operator into a Qiskit Operator.

    Arguments:
        of_operator (QubitOperator): a linear combination of Pauli strings as an Openfermion operator
        n (int): the size of the qubit register
        little_endian (bool): whether to revert use little endian ordering
    Returns:
        qiskit_operator (PauliSumOp): the original operator, represented in Qiskit
    """

    # If of_operator is an InteractionOperator, shape it into a FermionOperator
    if isinstance(of_operator, InteractionOperator):
        of_operator = get_fermion_operator(of_operator)

    # print(of_operator)
    if not n:
        n = count_qubits(of_operator)

    # print("N qubits: ",n)
    # Now use the Jordan Wigner transformation to map the FermionOperator into
    # a QubitOperator
    if isinstance(of_operator, FermionOperator):
        of_operator = jordan_wigner(of_operator)

    qiskit_operator = None

    # Iterate through the terms in the operator. Each is a Pauli string
    # multiplied by a coefficient
    for term in of_operator.get_operators():
        # print("==TERM==",term)
        if list(term.terms.keys())==[()]:
            # coefficient = term.terms[term.terms.keys()[0]]
            coefficient = term.terms[list(term.terms.keys())[0]]

            result = I
            # print("n", n)
            for _ in range(n-1):
                result = result ^ I

            qiskit_term = coefficient * result
            # print("empty qiskit term", qiskit_term)
        else:
            qiskit_term = to_qiskit_term(term, n, little_endian)
            # print("non empty qiskit term",qiskit_term)

        if qiskit_operator is None:
            qiskit_operator = qiskit_term
        else:
            qiskit_operator += qiskit_term

    return qiskit_operator

