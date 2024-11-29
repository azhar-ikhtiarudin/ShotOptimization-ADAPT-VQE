import copy
import numpy as np

from .adapt_data import AdaptData
from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2

from src.utilities import to_qiskit_operator

class AdaptVQE():
    """
        Main Class for ADAPT-VQE Algorithm
    """

    def __init__(self, pool, molecule, max_adapt_iter, max_opt_iter, grad_threshold=10**-8, vrb=False):
        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.vrb = vrb
        self.grad_threshold = grad_threshold

        self.data = None

        self.n = self.molecule.n_qubits
        self.fermionic_hamiltonian = self.molecule.get_molecular_hamiltonian()
        self.qubit_hamiltonian = jordan_wigner(self.fermionic_hamiltonian)
        self.exact_energy = self.molecule.fci_energy
        self.window = 1

        # Hartree Fock Reference State:
        self.ref_determinant = [ 1 for _ in range(self.molecule.n_electrons) ]
        self.ref_determinant += [ 0 for _ in range(self.fermionic_hamiltonian.n_qubits - self.molecule.n_electrons ) ]

        if self.vrb:
            print(". . . == ADAPT-VQE Settings == . . .")
            print("Fermionic Hamiltonian:", self.fermionic_hamiltonian)
            print("Qubit Hamiltonian:", self.qubit_hamiltonian)
            print("Hartree Fock Reference State:", self.ref_determinant)

    def run(self):
        if self.vrb: print("\nRun, Initializing . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()


    def initialize(self):
        self.initial_energy = self.evaluate_observable(self.qubit_hamiltonian) 
        # self.initial_energy = -1.25

        if not self.data:
            self.data = AdaptData(self.initial_energy, self.pool, self.exact_energy, self.n)
            self.indices = []
            self.coefficients = []
            self.old_coefficients = []
            self.old_gradients = []
        return


    def run_iteration(self):
        if self.vrb: print("ADAPT-VQE Run Iteration")

        # Gradient Screening
        finished, viable_candidates, viable_gradients, total_norm = ( self.start_iteration() )

        if finished: return finished
    
    def start_iteration(self):
        if self.vrb: print("\n\n. . . === ADAPT-VQE Iteration", self.data.iteration_counter + 1, "=== . . .")

        viable_candidates, viable_gradients, total_norm, max_norm = ( self.rank_gradients() )


    def rank_gradients(self, coefficients=None, indices=None):
        sel_gradients = []
        sel_indices = []
        total_norm = 0

        if self.vrb: print("-Pool Size:", self.pool.size)

        for index in range(self.pool.size):
            if self.vrb: print("\n--- Evaluating Gradient", index)

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            if self.vrb: print("-Gradient:", gradient)

            if np.abs(gradient) < self.grad_threshold: continue

            sel_gradients, sel_indices = self.place_gradient( gradient, index, sel_gradients, sel_indices )
            print("Selected Gradients:", sel_gradients)
            print("Selected Gradients Type:", type(sel_gradients))
            print("Parent Range:", self.pool.parent_range)
            if index not in self.pool.parent_range: 
                print("---", index, self.pool.parent_range)
                total_norm += gradient**2
                print("___total norm:", total_norm, "___gradient:", gradient**2)

        total_norm = np.sqrt(total_norm)

        if sel_gradients: max_norm = sel_gradients[0]
        else: max_norm = 0

        if self.vrb:
            print("Total gradient norm: {}".format(total_norm))
            print("Final Selected Indices:", sel_indices)
            print("Final Selected Gradients:", sel_gradients)
            print("Total Norm", total_norm)
            print("Max Norm", max_norm)
        
        return sel_indices, sel_gradients, total_norm, max_norm
    
    def eval_candidate_gradient(self, index, coefficients=None, indices=None):
        measurement = self.pool.get_grad_meas(index)

        if measurement is None:
            operator = self.pool.get_q_op(index)
            observable = commutator(self.qubit_hamiltonian, operator)
            
            if self.vrb: 
                print("Operator", self.pool.get_q_op(index))
                print("Observable", observable)
            
            self.pool.store_grad_meas(index, measurement)
        
        gradient = self.evaluate_observable(observable, coefficients, indices)

        return gradient

    def evaluate_observable(self, observable, coefficients=None, indices=None):
        qiskit_observable = to_qiskit_operator(observable)
        if self.vrb: print("\nQiskit Observable:", qiskit_observable)

        # Obtain Quantum Circuit
        qc = QuantumCircuit(self.n)
        print("\n",self.ref_determinant)

        for i, qubit in enumerate(self.ref_determinant):
            # print("i:",i,"qubit:",qubit)
            if qubit == 1: qc.x(i)
        
        print(qc)
        print("coefficients:",coefficients,", indices:", indices)
        print("FCI Energy", self.molecule.fci_energy)
        if indices is not None and coefficients is not None:
            qc = self.pool.get_circuit(indices, coefficients)
        
        if self.vrb: print("Quantum Circuit:", qc)

        estimator = EstimatorV2(backend=AerSimulator())
        job = estimator.run([(qc, qiskit_observable)])
        exp_vals = job.result()[0].data.evs
        print("EXPECTATION VALUES", exp_vals)
        return exp_vals
    
    def place_gradient(self, gradient, index, sel_gradients, sel_indices):

        i = 0

        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < self.grad_threshold:
                condition = self.break_gradient_tie(gradient, sel_gradient)
                if condition: break
            
            elif np.abs(gradient) - np.abs(sel_gradient) >= self.grad_threshold:
                break

            i += 1
        
        if i < self.window:
            sel_indices = sel_indices[:i] + [index] + sel_indices[i : self.window - 1]

            sel_gradients = (
                sel_gradients[:i] + [gradient] + sel_gradients[i : self.window - 1]
            )
        
        return sel_gradients, sel_indices

    def break_gradient_tie(self, gradient, sel_gradient):
        assert np.abs(np.abs(gradient) - np.abs(sel_gradient)) < self.grad_threshold

        if self.rand_degenerate:
            # Position before/after with 50% probability
            condition = np.random.rand() < 0.5
        else:
            # Just place the highest first even if the difference is small
            condition = np.abs(gradient) > np.abs(sel_gradient)

        return condition

