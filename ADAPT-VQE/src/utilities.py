
import re
import numpy as np
from openfermion import (
    count_qubits,
    FermionOperator,
    QubitOperator,
    get_fermion_operator,
    InteractionOperator,
    jordan_wigner,
)
from qiskit import QuantumCircuit
from openfermion.ops.representations import DiagonalCoulombHamiltonian, PolynomialTensor
from openfermion.ops.operators import FermionOperator, QubitOperator, BosonOperator, QuadOperator
import qiskit
from qiskit.qasm3 import dumps
from qiskit.quantum_info import Pauli, SparsePauliOp

I = SparsePauliOp("I")
X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")

def get_qasm(qc):
    """
    Converts a Qiskit QuantumCircuit to qasm.
    Args:
        qc (QuantumCircuit): a Qiskit QuantumCircuit

    Returns:
        qasm (str): the QASM string for this circuit
    """

    if int(qiskit.__version__[0]) >= 1:
        qasm = dumps(qc)
    else:
        qasm = qc.qasm()

    return qasm


def bfgs_update(hk, gfkp1, gfk, xkp1, xk):
    """
    Performs a BFGS update.

    Arguments:
        hk (np.ndarray): the previous inverse Hessian (iteration k)
        gfkp1 (np.ndarray): the new gradient vector (iteration k + 1)
        gfk (np.ndarray): the old gradient vector (iteration k)
        xkp1 (np.ndarray): the new parameter vector (iteration k + 1)
        xk (np.ndarray):  the old parameter vector (iteration k)

    Returns:
        hkp1 (np.darray): the new inverse Hessian (iteration k + 1)
    """

    gfkp1 = np.array(gfkp1)
    gfk = np.array(gfk)
    xkp1 = np.array(xkp1)
    xk = np.array(xk)

    n = len(xk)
    id_mat = np.eye(n, dtype=int)

    sk = xkp1 - xk
    yk = gfkp1 - gfk

    rhok_inv = np.dot(yk, sk)
    if rhok_inv == 0.:
        rhok = 1000.0
        print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    a1 = id_mat - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    a2 = id_mat - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    hkp1 = np.dot(a1, np.dot(hk, a2)) + (rhok * sk[:, np.newaxis] *
                                         sk[np.newaxis, :])

    return hkp1


def get_operator_qubits(operator):
    """
    Obtains the support of an operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        qubits (Set): List containing the indices of the qubits in which operator acts on non-trivially
    """
    qubits = set()

    for string in list(operator.terms.keys()):
        for qubit, pauli in string:
            if qubit not in qubits:
                qubits.add(qubit)

    return qubits


def remove_z_string(operator):
    """
    Removes the anticommutation string from Jordan-Wigner transformed excitations. This is equivalent to removing
    all Z operators.
    This function does not change the original operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        new_operator (Union[FermionOperator, QubitOperator]): the same operator, with Pauli-Zs removed
    """

    if isinstance(operator, QubitOperator):
        qubit_operator = operator
    else:
        qubit_operator = jordan_wigner(operator)

    new_operator = QubitOperator()

    for term in qubit_operator.get_operators():

        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]

        new_pauli = QubitOperator((), coefficient)

        for qubit, operator in pauli_string:
            if operator != 'Z':
                new_pauli *= QubitOperator((qubit, operator))

        new_operator += new_pauli

    return new_operator


def cnot_depth(qasm, n):
    """
    Counts the depth of a circuit on n qubits represented by a QASM string, considering only cx gates.
    Circuit must be decomposed into a cx + single qubit rotations gate set.

    Aguments:
        qasm (str): the QASM representation of the circuit
        n (int): the number of qubits
    Returns:
        The CNOT depth of the circuit
    """
    # n = int(re.search(r"(?<=q\[)[0-9]+(?=\])", qasm.splitlines()[2]).group())
    depths = [0 for _ in range(n)]

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]

        # Split line by spaces
        line_elems = line.split(" ")

        # First element is operation type
        op = line_elems[0]
        if op[:2] != "cx":
            continue

        # Next elements are qubits
        qubits = [
            int(re.search(r"[0-9]+", qubit_string).group())
            for qubit_string in line_elems[1:]
        ]

        max_depth = max([depths[qubit] for qubit in qubits])
        new_depth = max_depth + 1

        for qubit in qubits:
            depths[qubit] = new_depth

    return max(depths)


def cnot_count(qasm):
    """
    Counts the CNOTs in a circuit represented by a QASM string.
    """
    count = 0

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]
        line_elems = line.split(" ")
        op = line_elems[0]

        if op[:2] == "cx":
            count += 1

    return count




def qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 2:
        return double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)
    else:
        return single_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)


def double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.x(b)
    qc.x(d)
    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.h(b)
    qc.cx(a, b)
    qc.h(d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.h(c)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.h(d)
    qc.h(b)
    qc.rz(+np.pi / 2, c)
    qc.cx(a, c)

    qc.rz(-np.pi / 2, a)
    qc.rz(+np.pi / 2, c)
    qc.ry(+np.pi / 2, c)

    qc.x(b)
    qc.x(d)
    qc.cx(a, b)
    qc.cx(c, d)

    return qc

def normalize_op(operator):
    """
    Normalize Qubit or Fermion Operator by forcing the absolute values of the coefficients to sum to zero.
    This function modifies the operator.

    Arguments:
        operator (Union[FermionOperator,QubitOperator]): the operator to normalize

    Returns:
        operator (Union[FermionOperator,QubitOperator]): the same operator, now normalized0
    """

    if operator:
        coeff = 0
        for t in operator.terms:
            coeff_t = operator.terms[t]
            # coeff += np.abs(coeff_t * coeff_t)
            coeff += np.abs(coeff_t)

        # operator = operator/np.sqrt(coeff)
        operator = operator / coeff

    return operator

def get_hf_det(electron_number, qubit_number):
    """
    Get the Hartree Fock ket |1>|1>...|0>|0>.

    Arguments:
    electron_number (int): the number of electrons of the molecule.
    qubit_number (int): the number of qubits necessary to represent the molecule
      (equal to the number of spin orbitals we're considering active).

    Returns:
    reference_ket (list): a list of lenght qubit_number, representing the
      ket of the adequate computational basis state in big-endian ordering.
    """

    # Consider occupied the lower energy orbitals, until enough one particle
    # states are filled
    reference_ket = [1 for _ in range(electron_number)]

    # Consider the remaining orbitals empty
    reference_ket += [0 for _ in range(qubit_number - electron_number)]

    return reference_ket



def ket_to_vector(ket,little_endian=False):
    """
    Transforms a ket representing a basis state to the corresponding state vector.

    Arguments:
        ket (list): a list of length n representing the ket
        little_endian (bool): whether the input ket is in little endian notation

    Returns:
        state_vector (np.ndarray): the corresponding basis vector in the
            2^n dimensional Hilbert space
    """

    if little_endian:
        ket = ket[::-1]

    state_vector = [1]

    # Iterate through the ket, calculating the tensor product of the qubit states
    for i in ket:
        qubit_vector = [not i, i]
        state_vector = np.kron(state_vector, qubit_vector)

    return state_vector


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
    print("pauli string", pauli_string)
    for qubit_index, pauli in pauli_string:
        print("AA")
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
        print("--qiskit_op-1", qiskit_op)
        previous_index = qubit_index

    id_count = (n - previous_index - 1)
    print("BB", switch_endianness)
    if switch_endianness:
        print(id_count)
        for _ in range(id_count):
            print("--I", I)
            print("--qiskit_op-2", qiskit_op)
            qiskit_op = I ^ qiskit_op
    else:
        for _ in range(id_count):
            qiskit_op = qiskit_op ^ I
    
    print("coefficient", coefficient)
    print("qiskit_op", qiskit_op)

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

    print("N qubits: ",n)
    # Now use the Jordan Wigner transformation to map the FermionOperator into
    # a QubitOperator
    if isinstance(of_operator, FermionOperator):
        of_operator = jordan_wigner(of_operator)

    qiskit_operator = None

    # Iterate through the terms in the operator. Each is a Pauli string
    # multiplied by a coefficient
    for term in of_operator.get_operators():
        print("==TERM==",term)
        if list(term.terms.keys())==[()]:
            # coefficient = term.terms[term.terms.keys()[0]]
            coefficient = term.terms[list(term.terms.keys())[0]]

            result = I
            print("n", n)
            for _ in range(n-1):
                result = result ^ I

            qiskit_term = coefficient * result
            print("empty qiskit term", qiskit_term)
        else:
            qiskit_term = to_qiskit_term(term, n, little_endian)
            print("non empty qiskit term",qiskit_term)

        if qiskit_operator is None:
            qiskit_operator = qiskit_term
        else:
            qiskit_operator += qiskit_term

    return qiskit_operator
