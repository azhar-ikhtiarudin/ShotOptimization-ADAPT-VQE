
import abc
import numpy as np

from copy import copy
from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered, jordan_wigner, get_sparse_operator
from openfermion.transforms import freeze_orbitals

from .utilities import get_operator_qubits, remove_z_string, cnot_depth, cnot_count, qe_circuit, normalize_op

from scipy.sparse import csc_matrix, issparse, identity
from scipy.sparse.linalg import expm, expm_multiply

from qiskit import QuantumCircuit

class OpType:
    FERMIONIC = 0
    QUBIT = 1


class ImplementationType:
    SPARSE = 0
    QISKIT = 1


class PoolOperator(metaclass=abc.ABCMeta):

    def __init__(self, operator, n, tag, frozen_orbitals=[], cnots=None, cnot_depth=None, parents=None,
                 source_orbs=None, target_orbs=None, ceo_type=None):
        """
        Arguments:
            operator(Union[FermionOperator,QubitOperator]: the operator we want to represent
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            tag (int): number identifying position in pool
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            cnots (int): number of CNOTs in the circuit implementation of this operator
            cnot_depth (int): CNOT depth in the circuit implementation of this operator
            parents (list): indices of operators that this one derives from (in the case of CEOs, where operator is
                linear combination of parents)
            source_orbs (list): spin-orbitals from which the operator removes fermions
            target_orbs (list): spin-orbitals to which the operator adds fermions
            ceo_type (str): "sum" or "diff", defining the type of OVP-CEO when applicable

        Note: Operator may be modified by class methods!
        If this were not desired, we could do self._f_operator = operator * 1. This creates a new copy of the operator.
        """

        if isinstance(operator, FermionOperator):
            if frozen_orbitals:
                self._f_operator = freeze_orbitals(operator, frozen_orbitals)
            else:
                self._f_operator = operator

            self._q_operator = None
            self.op_type = OpType.FERMIONIC

        elif isinstance(operator, QubitOperator):
            self._f_operator = None
            self._q_operator = operator
            self.op_type = OpType.QUBIT
            self.cnots = cnots
            self.cnot_depth = cnot_depth
            self.parents = parents

        else:
            raise TypeError("Expected Fermion or QubitOperator, not {}."
                            .format(type(operator).__name__))

        self.qubits = get_operator_qubits(operator)
        self.n = n
        self.tag = tag
        self.coef = None
        self.imp_operator = None  # implemented version (e.g. Qiskit Operator)
        self.exp_operator = None  # exponential version (e.g. trotter circuit)
        self.grad_meas = None  # gradient observable
        self.twin_string_ops = []  # operators in the same pool with the exact same Pauli strings
        self.source_orbs = source_orbs
        self.target_orbs = target_orbs
        self.ceo_type = ceo_type

    def __str__(self):

        return self.operator.__str__()

    def __eq__(self, other):

        if isinstance(other, PoolOperator):
            return (self.operator == other.operator or
                    self.operator == - other.operator)

        return False

    def arrange(self):
        """
        Arrange self.
        If self is a fermionic operator $\tau$, it will be made into a proper
        anti-hermitian pool operator $\tau$ - hc($\tau$) and normal-ordered.
        Both fermionic and qubit operators are normalized also.

        Return value: True if the operator is nontrivial, true if it's trivial

        This does not change the state.
        """

        if self.op_type == OpType.FERMIONIC:
            # Subtract hermitian conjugate to make the operator anti Hermitian
            self._f_operator -= hermitian_conjugated(self._f_operator)

            # Normal order the resulting operator so that we have a consistent ordering
            self._f_operator = normal_ordered(self._f_operator)

        if not self.operator.many_body_order():
            # Operator acts on 0 qubits; discard
            return False

        self.normalize()

        return True

    def normalize(self):
        """
        Normalize self, so that the sum of the absolute values of coefficients is one.
        """

        self._f_operator = normalize_op(self._f_operator)
        self._q_operator = normalize_op(self._q_operator)

    def create_qubit(self):
        """
        Create a qubit version of the fermion operator.
        """

        if not self._q_operator:
            self._q_operator = normalize_op(jordan_wigner(self._f_operator))

    def create_sparse(self):
        """
        Obtain sparse matrix representing the space, in the proper dimension (might be higher than the effective
        dimension of operator)
        """
        self.imp_operator = get_sparse_operator(self.q_operator, self.n)

    @property
    def f_operator(self):
        return self._f_operator

    @property
    def q_operator(self):
        if not self._q_operator:
            self.create_qubit()
        return self._q_operator

    @property
    def operator(self):
        if self.op_type == OpType.QUBIT:
            return self._q_operator
        if self.op_type == OpType.FERMIONIC:
            return self._f_operator



class OperatorPool(metaclass=abc.ABCMeta):
    name = None

    def __init__(self, molecule=None, frozen_orbitals=[], n=None, source_ops=None):
        """
        Arguments:
            molecule (PyscfMolecularData): the molecule for which we will use the pool
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            source_ops (list): the operators to generate the pool from, if tiling.
        """

        if self.name is None:
            raise NotImplementedError("Subclasses must define a pool name.")

        if self.op_type == OpType.QUBIT:
            self.has_qubit = True
        else:
            self.has_qubit = False

        self.frozen_orbitals = frozen_orbitals
        self.molecule = molecule
        self.source_ops = source_ops

        if molecule is None:
            assert n is not None
            self.n = n
        else:
            self.n_so = molecule.n_orbitals  # Number of spatial orbitals
            self.n = molecule.n_qubits - len(frozen_orbitals)  # Number of qubits = 2*n_so

        self.operators = []
        self._ops_on_qubits = {}

        self.create_operators()
        self.eig_decomp = [None for _ in range(self.size)]
        self.squared_ops = [None for _ in range(self.size)]

        # Get range of parent doubles - i.e., double excitations from which pool is generated by taking sums/differences
        # Only applicable if we have CEO pool.
        for i in range(self.size):
            if len(self.get_qubits(i)) == 4:
                break
        if self.name[:3] in ["DVG", "DVE"]:
            self.parent_range = range(i, self.parent_pool.size)
        else:
            self.parent_range = []

    def __add__(self, other):

        assert isinstance(other, OperatorPool)
        assert self.n == other.n
        assert self.op_type == other.op_type

        pool = copy(self)
        pool.operators = copy(self.operators)

        for operator in other.operators:
            if operator not in pool.operators:
                pool.operators.append(operator)

        pool.name = pool.name + "_+_" + other.name
        pool.eig_decomp = pool.eig_decomp + other.eig_decomp
        pool.couple_exchanges = self.couple_exchanges or other.couple_exchanges

        return pool

    def __str__(self):

        if self.op_type == OpType.QUBIT:
            type_str = "Qubit"
        if self.op_type == OpType.FERMIONIC:
            type_str = "Fermionic"

        text = f"{type_str} pool with {self.size} operators\n"

        for i, operator in enumerate(self.operators):
            text += f"{i}:\n{str(operator)}\n\n"

        return text

    def add_operator(self, new_operator, cnots=None, cnot_depth=None, parents=None, source_orbs=None, target_orbs=None,
                     ceo_type=None):
        """
        Arguments:
            new_operator (Union[PoolOperator,FermionOperator,QubitOperator]): operator to add to pool
            cnots (int): number of CNOTs in the circuit implementation of this operator
            cnot_depth (int): CNOT depth in the circuit implementation of this operator
            parents (list): indices of operators that this one derives from (in the case of CEOs, where operator is a
                linear combination of parents)
            source_orbs (list): spin-orbitals from which the operator removes fermions
            target_orbs (list): spin-orbitals to which the operator adds fermions
            ceo_type (str): "sum" or "diff", defining the type of OVP-CEO when applicable
        """

        if not isinstance(new_operator, PoolOperator):
            new_operator = PoolOperator(new_operator,
                                        self.n,
                                        self.size,
                                        self.frozen_orbitals,
                                        cnots,
                                        cnot_depth,
                                        parents,
                                        source_orbs,
                                        target_orbs,
                                        ceo_type)

        is_nontrivial = new_operator.arrange()

        if is_nontrivial and new_operator not in self.operators:
            self.operators.append(new_operator)
            position = len(self.operators) - 1
            return position

        return None

    @property
    def imp_type(self):
        return self._imp_type

    @imp_type.setter
    def imp_type(self, imp_type):
        if imp_type not in [ImplementationType.SPARSE]:
            raise ValueError("Argument isn't a valid implementation type.")

        self._imp_type = imp_type

    @abc.abstractmethod
    def create_operators(self):
        """
        Fill self.operators list with PoolOperator objects
        """
        pass

    @abc.abstractmethod
    def get_circuit(self, coefficients, indices):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments, as a Qiskit QuantumCircuit.
        Arguments:
            indices (list)
            coefficients (list)
        """
        pass

    def create_sparse(self):

        for operator in self.operators:
            operator.create_sparse()

    def create_eig_decomp(self, index):
        """
        Create eigendecomposition for operator represented by the given index (which identifies its place in the pool).
        Having the eigendecomposition facilitates implementing the exponential of the operator, because you can
        simply do a basis rotation, exponentiate a diagonal matrix, and revert the basis rotation.
        The exponential of a diagonal matrix is easy to obtain because you simply exponentiate the diagonal entries.
        Once you have the eigendecomposition, the calculations become much faster, because you do two matrix
        multiplications instead of one matrix exponentiation (which is significantly more complex).
        However, it might take quite some time to create the eigendecomposition for a complete pool. This becomes
        intractable for 14 qubits or more.
        """

        if self.eig_decomp[index] is None:
            print("Diagonalizing operator...")
            dense_op = self.get_imp_op(index).todense()
            # eigh is for Hermitian matrices, H is skew-Hermitian. Multiply -1j, undo later
            hermitian_op = -1j * dense_op
            w, v = np.linalg.eigh(hermitian_op)  # hermitian_op = v * diag(w) * inv(v)
            v[abs(v) < 1e-16] = 0
            v = csc_matrix(v)
            self.eig_decomp[index] = 1j * w, v

    def create_eig_decomps(self):
        """
        Create eigendecomposition for operator represented by the given index (which identifies its place in the pool).
        Having the eigendecomposition facilitates implementing the exponential of the operator, because you can
        simply do a basis rotation, exponentiate a diagonal matrix, and revert the basis rotation.
        The exponential of a diagonal matrix is easy to obtain because you simply exponentiate the diagonal entries.
        Once you have the eigendecomposition, the calculations become much faster, because you do two matrix
        multiplications instead of one matrix exponentiation (which is significantly more complex).
        However, it might take quite some time to create the eigendecomposition for a complete pool. This becomes
        intractable for 14 qubits or more.
        """

        for index in range(self.size):
            self.create_eig_decomp(index)

    def get_op(self, index):
        """
        Returns the operator specified by its index in the pool.
        """

        if self.op_type == OpType.FERMIONIC:
            return self.get_f_op(index)
        else:
            return self.get_q_op(index)

    def get_qubits(self, index):
        """
        Returns list of qubits in which the operator specified by this index acts non trivially.
        """
        return self.operators[index].qubits

    def get_parents(self, index):
        """
        Applicable only to CEO operators.
        Returns the QEs from which the operator derives (by taking linear combination).
        """
        return self.operators[index].parents

    def get_ops_on_qubits(self, qubits):
        """
        Returns the indices of the operators in the pool that act on the given qubits.
        """

        # Use this instead of directly accessing self._ops_on_qubits - the key must be sorted, you know you'll forget
        if not self._ops_on_qubits:
            raise ValueError("Operators have not been associated to qubits in this pool.")
        return self._ops_on_qubits[str(sorted(qubits))]

    def get_twin_ops(self, index):
        """
        Returns the indices of the operators in the pool that act on the same qubits as the operator identified by index
        """
        return self.operators[index].twin_string_ops

    def get_imp_op(self, index):
        """
        Returns implemented version of operator (depends on implementation type).
        """

        if self.operators[index].imp_operator is None:
            # print("ImplementationType.SPARSE", ImplementationType.SPARSE)
            # print("self.imp_type", self.imp_type)
            if self.imp_type == ImplementationType.SPARSE:
                self.operators[index].create_sparse()
            else:
                raise AttributeError("PoolOperator does not have imp_operator attribute because an implementation type "
                                     "hasn't been set for this pool. "
                                     "Please choose an implementation by setting pool.imp_type.")

        return self.operators[index].imp_operator

    def get_f_op(self, index):
        """
        Get fermionic operator labeled by index.
        """
        return self.operators[index].f_operator

    def get_q_op(self, index):
        """
        Get qubit operator labeled by index.
        """
        return self.operators[index].q_operator

    def get_exp_op(self, index, coefficient=None):
        """
        Get exponential of operator labeled by index.
        """
        if self.op_type == ImplementationType.SPARSE:
            return expm(coefficient * self.operators[index].imp_operator)
        else:
            raise ValueError

    def square(self, index):
        """
        Get square of operator labeled by index.
        It can be useful to store the value to make the computation faster.
        """

        op = self.get_imp_op(index)
        self.squared_ops[index] = op.dot(op)

        return self.squared_ops[index]

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.

        Arguments:
            coefficient (float)
            index (int)
        """
        assert self.op_type == ImplementationType.SPARSE

        if self.eig_decomp[index] is None:
            return expm(coefficient * self.operators[index].imp_operator)
        else:
            diag, unitary = self.eig_decomp[index]
            exp_diag = np.exp(coefficient * diag)
            exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
            return unitary.dot(np.multiply(exp_diag, unitary.T.conjugate().todense()))

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """

        assert self.imp_type == ImplementationType.SPARSE

        if self.eig_decomp[index] is None:
            if not issparse(other):
                other = csc_matrix(other)
            return expm_multiply(coefficient * self.operators[index].imp_operator, other)
        else:
            if issparse(other):
                other = other.todense()
            diag, unitary = self.eig_decomp[index]
            exp_diag = np.exp(coefficient * diag)
            exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
            m = unitary.T.conjugate().dot(other)
            m = np.multiply(exp_diag, m)
            m = unitary.dot(m)
            m = m.real
            return m




    # def expm_matrix(self, coefficient, index):
    #     """
    #     Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
    #     If an eigendecomposition of the operator exists, it will be used for increased efficiency.

    #     Arguments:
    #         coefficient (float)
    #         index (int)
    #     """
    #     # assert self.op_type == ImplementationType.SPARSE

    #     if self.eig_decomp[index] is None:
    #         return expm(coefficient * self.operators[index].create_sparse())
    #     else:
    #         diag, unitary = self.eig_decomp[index]
    #         exp_diag = np.exp(coefficient * diag)
    #         exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
    #         return unitary.dot(np.multiply(exp_diag, unitary.T.conjugate().todense()))

    # def expm_mult_matrix(self, coefficient, index, other):
    #     """
    #     Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
    #     another pool operator (indexed "other").
    #     If an eigendecomposition of the operator exists, it will be used for increased efficiency.

    #     Arguments:
    #         coefficient (float)
    #         index (int)
    #         other (csc_matrix)
    #     """

    #     # assert self.imp_type == ImplementationType.SPARSE

    #     if self.eig_decomp[index] is None:
    #         if not issparse(other):
    #             other = csc_matrix(other)
    #         return expm_multiply(coefficient * self.operators[index].create_sparse(), other)
    #     else:
    #         if issparse(other):
    #             other = other.todense()
    #         diag, unitary = self.eig_decomp[index]
    #         exp_diag = np.exp(coefficient * diag)
    #         exp_diag = exp_diag.reshape(exp_diag.shape[0], 1)
    #         m = unitary.T.conjugate().dot(other)
    #         m = np.multiply(exp_diag, m)
    #         m = unitary.dot(m)
    #         m = m.real
    #         return m

    def get_cnots(self, index):
        """
        Obtain number of CNOTs required in the circuit implementation of the operator labeled by index.
        If index is a list, it must represent a MVP-CEO.
        """

        if isinstance(index,list):

            # Make sure all operators are qubit excitations acting on the same qubits. If they are, the number of CNOTs
            #required in the circuit implementation is the same regardless of the number of operators
            op_qubits = [self.get_qubits(i) for i in index]
            assert all(qubits == op_qubits[0] for qubits in op_qubits)
            assert all([i in self.parent_range for i in index])
            index = index[0]

        return self.operators[index].cnots

    def get_cnot_depth(self, index):
        """
        Obtain CNOT depth of the circuit implementation of the operator labeled by index.
        """
        return self.operators[index].cnot_depth

    def get_grad_meas(self, index):
        """
        Obtain observable corresponding to the (energy) gradient of the operator labeled by index.
        """
        return self.operators[index].grad_meas

    def store_grad_meas(self, index, measurement):
        """
        Set the observable corresponding to the (energy) gradient of the operator labeled by index.
        """
        self.operators[index].grad_meas = measurement

    @abc.abstractproperty
    def op_type(self):
        """
        Type of pool (qubit/fermionic).
        """
        pass

    @property
    def size(self):
        """
        Number of operators in pool.
        """
        return len(self.operators)

    @property
    def exp_operators(self):
        """
        List of pool operators, in their exponential versions.
        """
        return [self.get_exp_op(i) for i in range(self.size)]

    @property
    def imp_operators(self):
        """
        List of pool operators, in their implemented versions.
        """
        return [self.get_imp_op(i) for i in range(self.size)]

    def cnot_depth(self, coefficients, indices):
        """
        Obtain CNOT depth of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return cnot_depth(circuit.qasm())

    def depth(self, coefficients, indices):
        """
        Obtain depth of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return circuit.depth

    def cnot_count(self, coefficients, indices):
        """
        Obtain CNOT count of the circuit implementation of the ansatz represented by input lists of coefficients
        and pool operator indices.
        """
        circuit = self.get_circuit(coefficients, indices)
        return cnot_count(circuit.qasm())


class QE(OperatorPool):
    """
    Pool consisting of qubit excitations, which are obtained by removing the Z strings from fermionic generalized
    single and double excitations. Instead of building a GSD pool first, we create the operators by iterating through
    combinations of indices we know are associated with valid excitations. This is more efficient than QE1.
    """

    name = "QE"

    def __init__(self,
                 molecule=None,
                 couple_exchanges=False,
                 frozen_orbitals=[],
                 n=None,
                 source_ops=None):
        """
        Arguments:
            molecule (PyscfMolecularData): the molecule for which we will use the pool
            couple_exchanges (bool): whether to add all qubit excitations with nonzero gradient acting on the same
                qubits when a given double qubit excitation is added to the ansatz. If this flag is set to True,
                the pool will correspond to the MVP-CEO pool when used in ADAPT-VQE.
            frozen_orbitals (list): indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            n (int): number of qubits the operator acts on. Note that this includes identity operators - it's dependent
            on system size, not operator support
            source_ops (list): the operators to generate the pool from, if tiling.
        """

        self.couple_exchanges = couple_exchanges

        if couple_exchanges:
            self.name = "MVP_CEO"

        super().__init__(molecule, frozen_orbitals, n=n, source_ops=source_ops)

    def create_operators(self):
        """
        Create pool operators and insert them into self.operators (list).
        """

        self.create_singles()
        self.create_doubles()

    def create_singles(self):
        """
        Create one-body qubit excitations.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                if (p + q) % 2 == 0:
                    f_operator = FermionOperator(((p, 1), (q, 0)))
                    f_operator -= hermitian_conjugated(f_operator)
                    f_operator = normal_ordered(f_operator)

                    q_operator = remove_z_string(f_operator)
                    pos = self.add_operator(q_operator, cnots=2, cnot_depth=2,
                                            source_orbs=[q], target_orbs=[p])
                    self._ops_on_qubits[str([p, q])] = [pos]

    def create_doubles(self):
        """
        Create two-body qubit excitations.
        """

        for p in range(0, self.n):

            for q in range(p + 1, self.n):

                for r in range(q + 1, self.n):

                    for s in range(r + 1, self.n):

                        if (p + q + r + s) % 2 != 0:
                            continue

                        # If aaaa or bbbb, all three of the following ifs apply, but there are only 3 distinct operators
                        # In the other cases, only one of the ifs applies, 2 distinct operators

                        new_positions = []
                        if (p + r) % 2 == 0:
                            # pqrs is abab or baba, or aaaa or bbbb

                            f_operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
                            f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

                            f_operator_1 -= hermitian_conjugated(f_operator_1)
                            f_operator_2 -= hermitian_conjugated(f_operator_2)

                            f_operator_1 = normal_ordered(f_operator_1)
                            f_operator_2 = normal_ordered(f_operator_2)

                            q_operator_1 = remove_z_string(f_operator_1)
                            q_operator_2 = remove_z_string(f_operator_2)

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[r, s], target_orbs=[p, q])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[p, s], target_orbs=[q, r])

                            new_positions += [pos1, pos2]

                        if (p + q) % 2 == 0:
                            # aabb or bbaa, or aaaa or bbbb

                            f_operator_1 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))
                            f_operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

                            f_operator_1 -= hermitian_conjugated(f_operator_1)
                            f_operator_2 -= hermitian_conjugated(f_operator_2)

                            f_operator_1 = normal_ordered(f_operator_1)
                            f_operator_2 = normal_ordered(f_operator_2)

                            q_operator_1 = remove_z_string(f_operator_1)
                            q_operator_2 = remove_z_string(f_operator_2)

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[q, s], target_orbs=[p, r])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[p, s], target_orbs=[q, r])

                            new_positions += [pos1, pos2]

                        if (p + s) % 2 == 0:
                            # abba or baab, or aaaa or bbbb

                            f_operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
                            # f_operator_2 = FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)))
                            f_operator_2 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))

                            f_operator_1 -= hermitian_conjugated(f_operator_1)
                            f_operator_2 -= hermitian_conjugated(f_operator_2)

                            f_operator_1 = normal_ordered(f_operator_1)
                            f_operator_2 = normal_ordered(f_operator_2)

                            q_operator_1 = remove_z_string(f_operator_1)
                            q_operator_2 = remove_z_string(f_operator_2)

                            pos1 = self.add_operator(q_operator_1, cnots=13, cnot_depth=11,
                                                     source_orbs=[r, s], target_orbs=[p, q])

                            pos2 = self.add_operator(q_operator_2, cnots=13, cnot_depth=11,
                                                     source_orbs=[q, s], target_orbs=[p, r])

                            new_positions += [pos1, pos2]

                        new_positions = [pos for pos in new_positions if pos is not None]
                        self._ops_on_qubits[str([p, q, r, s])] = new_positions
                        if self.couple_exchanges:
                            for pos1, pos2 in itertools.combinations(new_positions, 2):
                                self.operators[pos1].twin_string_ops.append(pos2)
                                self.operators[pos2].twin_string_ops.append(pos1)

    @property
    def op_type(self):
        return OpType.QUBIT

    def expm(self, coefficient, index):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient.
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
            than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
        """
        if self.eig_decomp[index] is not None:
            return super().expm(index, coefficient)
        op = self.get_imp_op(index)
        n, n = op.shape
        exp_op = identity(n) + np.sin(coefficient) * op + (1 - np.cos(coefficient)) * self.square(index)
        return exp_op

    def expm_mult(self, coefficient, index, other):
        """
        Calculates the exponential of the operator defined by index, when multiplied by the coefficient, multiplying
        another pool operator (indexed "other").
        If an eigendecomposition of the operator exists, it will be used for increased efficiency.
        Otherwise, a trigonometric formula leveraging the structure of the operators is used. This is quite faster
        than using generic matrix exponentiation methods.

        Arguments:
            coefficient (float)
            index (int)
            other (csc_matrix)
        """
        if self.eig_decomp[index] is not None:
            return super().expm_mult(coefficient, index, other)
        '''
        exp_op = self.expm(index,coefficient)
        m = exp_op.dot(other)
        '''
        # It's faster to do product first, then sums; this way we never do matrix-matrix operations, just matrix-vector
        op = self.get_imp_op(index)
        m = op.dot(other)
        # In the following we can use self.square(index).dot(ket) instead of op.dot(m). But that's actually slightly
        # slower even if self.square(index) was already stored and we don't have to calculate it
        m = other + np.sin(coefficient) * m + (1 - np.cos(coefficient)) * op.dot(m)

        return m

    # def get_circuit(self, indices, coefficients, parameters):
    #     """
    #     Returns the circuit corresponding to the ansatz defined by the arguments.
    #     Function for the QE pool only.
    #     """

    #     circuit = QuantumCircuit(self.n)

    #     for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
    #         operator = self.operators[index]
    #         source_orbs = operator.source_orbs
    #         target_orbs = operator.target_orbs
    #         qc = qe_circuit(source_orbs, target_orbs, parameters[i], self.n, big_endian=False)

    #         circuit = circuit.compose(qc)
    #         circuit.barrier()

    #     return circuit
    
    def get_circuit(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for the QE pool only.
        """

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit


    def get_parameterized_circuit(self, indices, coefficients, parameters):

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficients) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, parameters[i], self.n, big_endian=False)
            circuit = circuit.compose(qc)
            circuit.barrier()
        
        return circuit
    




    def get_circuit_unparameterized(self, indices, coefficients):
        """
        Returns the circuit corresponding to the ansatz defined by the arguments.
        Function for the QE pool only.
        """

        circuit = QuantumCircuit(self.n)

        for i, (index, coefficient) in enumerate(zip(indices, coefficients)):
            operator = self.operators[index]
            source_orbs = operator.source_orbs
            target_orbs = operator.target_orbs
            qc = qe_circuit(source_orbs, target_orbs, coefficient, self.n, big_endian=True)

            circuit = circuit.compose(qc)
            circuit.barrier()

        return circuit
