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

        if self.vrb:
            print(". . . == ADAPT-VQE Settings == . . .")
            print("Fermionic Hamiltonian:", self.fermionic_hamiltonian)
            print("Qubit Hamiltonian:", self.qubit_hamiltonian)

    def run(self):
        if self.vrb: print("\nRun, Initializing . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()


    def initialize(self):
        # self.initial_energy = self.evaluate_energy() 
        self.initial_energy = -1.25

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
        if self.vrb: print("ADAPT-VQE Iteration",self.data.iteration_counter + 1)

        viable_candidates, viable_gradients, total_norm, max_norm = ( self.rank_gradients() )


    def rank_gradients(self, coefficients=None, indices=None):
        sel_gradients = []
        sel_indices = []
        total_norm = []

        for index in range(self.pool.size):
            if self.vrb: print("--Pool Size:", self.pool.size)

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            if self.vrb: print("--Gradient:", gradient)

            if np.abs(gradient) < self.grad_threshold: continue

            sel_gradients, sel_indices = self.place_gradient( gradient, index, sel_gradients, sel_indices )

            if index not in self.pool.parent_range: total_norm += gradient**2

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

    def evaluate_observable(self, observable, coefficients, indices):
        qiskit_observable = to_qiskit_operator(observable)
        if self.vrb: print("\nQiskit Observable:", qiskit_observable)

        estimator = EstimatorV2(backend=AerSimulator())
        job = estimator.run([(qc, qiskit_observable)])
        exp_vals = job.result()[0].data.evs
        return exp_vals




