
import numpy as np

from scipy.sparse import csc_matrix
from copy import deepcopy

from src.utilities import get_hf_det, ket_to_vector
from src.sparse_tools import get_sparse_operator 
from src.pools import ImplementationType

from .adapt_data import AdaptData

class AdaptVQE():

    def __init__(self,
                 pool,
                 molecule,
                 max_adapt_iter,
                 max_opt_iter,
                 verbose,
                 candidates=1,
                 threshold=0.1
                 ):
        
        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.verbose = verbose
        self.candidates = candidates
        self.threshold = threshold

        self.initialize_hamiltonian()
        self.file_name = (
            f"{self.molecule.description}_r={self.molecule.geometry[1][1][2]}"
        )
        self.gradients = np.array(())
        self.data = None
        self.set_window()

        # Matrix based
        self.state = self.sparse_ref_state
        self.ref_state = self.sparse_ref_state
        self.pool.imp_type = ImplementationType.SPARSE
        # self.energy_meas = self.observable_to_measurement(self.hamiltonian)
    
    def initialize_hamiltonian(self):

        if self.molecule is not None:
            # Initialize hamiltonian from molecule
            # self.n = self.molecule.n_qubits - len(self.frozen_orbitals)
            self.n = self.molecule.n_qubits
            # self.molecule.n_electrons -= len(self.frozen_orbitals)

            # Hartree Fock Reference State:
            self.ref_det = get_hf_det(self.molecule.n_electrons, self.n)

            self.sparse_ref_state = csc_matrix(
                ket_to_vector(self.ref_det), dtype = complex
                ).transpose()
            
            hamiltonian = self.molecule.get_molecular_hamiltonian()
            self.exact_energy = self.molecule.fci_energy
        
        if self.verbose:
            print("HF State:", self.ref_det)
            print("N Qubits:", self.n)
            print("Hamiltonian:", hamiltonian)
            print("Exact Energy:", self.exact_energy)

        self.hamiltonian = get_sparse_operator(hamiltonian, self.n)



    def load(self, previous_data=None, eig_decomp=None):
        
        if previous_data is not None:
            description_other = previous_data.file_name.split(
                str(previous_data.iteration_counter) + "i"
            )
            description_self = self.file_name.split(str(self.max_adapt_iter) + "i")

            # Description must match except iteration number
            assert description_self == description_other

            self.data = deepcopy(previous_data)
            self.data.file_name = self.file_name

            # Set current state to the last iteration of the loaded data
            self.indices = self.data.evolution.indices[-1]
            self.coefficients = self.data.evolution.coefficients[-1]
            self.gradients = self.data.evolution.gradients[-1]
            self.inv_hessian = self.data.evolution.inv_hessians[-1]

            # Calculate current state
            self.state = self.compute_state()

            print("Loaded indices: ", self.indices)
            print("Loaded coefficients: ", self.coefficients)


    def print_settings(self):
        pass

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        if coefficients is None or indices is None:
            # No ansatz provided, return current state
            if ref_state is not None:
                raise ValueError("Resulting state is just input reference state.")
            if coefficients is not None or indices is not None:
                raise ValueError("Cannot provide only coefficients or only indices.")
            state = self.state
        else:
            state = self.compute_state(coefficients, indices, ref_state)
        
        return state
    
    def compute_state(self, coefficients=None, indices=None, ref_state=None, bra=False):
        if indices is None:
            indices = self.indices
        if coefficients is None:
            coefficients = self.coefficients
        
        if ref_state is None:
            ref_state = self.sparse_ref_state
        state = ref_state.copy()

        if bra:
            coefficients = [-c for c in reversed(coefficients)]
            indices = reversed(indices)
        
        for coefficient, index in zip(coefficients, indices):
            state = self.pool.exp_mult(coefficient, index, state)
        if bra:
            state = state.transpose().conj()
        
        return state
    
    def evaluate_observable(self, 
                            observable,
                            coefficients=None,
                            indices=None,
                            ref_state=None,
                            orb_params=None
                            ):
         ket = self.get_state(coefficients, indices, ref_state)
         bra = ket.transpose().conj()
         exp_value = (bra.dot(observable.dot(ket)))[0,0].real

         print("\nEvaluating Observable . . .")
         print("ket:", ket)
         print("bra:", bra)
         print("obs:\n", observable)
         print("res:", exp_value, '\n')

         return exp_value
    
    def run(self):
        # Run Full ADAPT-VQE Algorithm
        self.initialize()
        
        finished = False
        print("First Run, finished=", finished)
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()

        # End of the loop
        if not finished:
            pass

        if finished:
            print("Converged!")
            error = self.energy - self.exact_energy
        else:
            print("Maximum iteration reached before converged!")
    
    def run_iteration(self):
        # Run one Iteration of the algorithm
        print("=== Run Iteration ===")
        finished, viable_candidates, viable_gradients, total_norm = (
            self.start_iteration()
        )

        if finished:
            return finished # already converged
        
        while viable_candidates:
            energy, g, viable_candidates, viable_gradients = self.grow_and_update(
                viable_candidates, viable_gradients
            )

        if energy is None:
            energy = self.optimize(g)
        
        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished
    
    def initialize(self):
        initial_energy = self.evaluate_observable(self.hamiltonian)
        self.energy = initial_energy

        if not self.data:
            self.data = AdaptData(
                initial_energy,
                self.pool,
                self.sparse_ref_state,
                self.file_name,
                self.exact_energy,
                self.n
            )
        else:
            assert self.energy - self.data.evolution.energies[-1] < 10**-12
        
        print("Initial Energy: ", initial_energy)

        return
    
    def start_iteration(self):
        print(f"\n*** ADAPT-VQE Iteration {self.data.iteration_counter + 1} ***\n")
        
        viable_candidates, viable_gradients, total_norm, max_norm = (
            self.rank_gradients()
        )

        print("viable candidates", viable_candidates)
        print("viable gradients", viable_gradients)

        finished = self.probe_termination(total_norm, max_norm)

        if finished:
            return finished, viable_candidates, viable_gradients, total_norm
        
        self.iteration_nfevs = []
        self.iteration_ngevs = []
        self.iteration_nits = []
        self.iteration_sel_gradients = []
        self.iteration_qubits = (set())

        return finished, viable_candidates, viable_gradients, total_norm
        
        

    
    def rank_gradients(self, coefficients=None, indices=None):
        
        sel_gradients = []
        sel_indices = []
        total_norm = 0
        print("--Pool Size: ", self.pool.size)

        for index in range(self.pool.size):
            # print("Current Operator: ", self.pool[index])
            print("Gradient idx, coeffs:", index, coefficients, indices)
            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            print("Gradient Result for Pool Index", index, ":", gradient)
            # gradient = self.penalize_gradient(gradient, index)

            if np.abs(gradient) < 10**-8:
                continue

            print("Initial Selected Gradients:", sel_gradients)
            print("Initial Selected Indices:", sel_indices)
            print("Initial Gradient:", gradient)
            print("Initial Index:", index)

            sel_gradients, sel_indices = self.place_gradient(
                gradient, index, sel_gradients, sel_indices
            )

            print("After Place Selected Gradients:", sel_gradients)
            print("After Place Selected Indices:", sel_indices)

            if index not in self.pool.parent_range:
                total_norm += gradient**2
            
        total_norm = np.sqrt(total_norm)

        if sel_gradients:
            max_norm = sel_gradients[0]
        else:
            max_norm = 0
        
        print("Final Selected Indices:", sel_indices)
        print("Final Selected Gradients:", sel_gradients)
        print("Total Norm", total_norm)
        print("Max Norm", max_norm)
        
        return sel_indices, sel_gradients, total_norm, max_norm
    
    def eval_candidate_gradient(self, index, coefficients=None, indices=None):
        
        measurement = self.pool.get_grad_meas(index)
        print("Measurement (get_grad_meas): ", measurement)

        if measurement is None:
            operator = self.pool.get_imp_op(index)
            print("operator", operator)
            observable = 2 * self.hamiltonian @ operator

            # measurement = self.observable_to_measurement(observable)
            # self.pool.store_grad_meas(index, observable)
        
        gradient = self.evaluate_observable(observable, coefficients, indices)

        return gradient
    
    def place_gradient(self, gradient, index, sel_gradients, sel_indices):
        i = 0
        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient) < 10**-8):
                condition = self.break_gradient_tie(gradient, sel_gradient)
                if condition: 
                    break
            elif np.abs(gradient) - np.abs(sel_gradient) >= 10**-8:
                break
            i = i + 1

        if i < self.window:
            sel_indices = sel_indices[:i] + [index] + sel_indices[i : self.window-1]
            sel_gradients = (
                sel_gradients[:i] + [gradient] + sel_gradients[i:self.window-1]
            )

        return sel_gradients, sel_indices

    def set_window(self):
        self.window = self.candidates
    
    def break_gradient_tie(self, gradient, sel_gradient):
        assert np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8

        if self.rand_degenerate:
            condition = np.random.rand() < 0.5
        else:
            condition = np.abs(gradient) > np.abs(sel_gradient)
        
        return condition
    
    def probe_termination(self, total_norm, max_norm):
        
        finished = False

        if total_norm < self.threshold and self.convergence_criterion == 'total_g_norm':
            self.converged()
            finished = True
        
        if max_norm < self.threshold and self.convergence_criterion == "max_g":
            self.converged()
            finished = True
        
        return finished



    # def penalize_gradient(self, gradient, index):
    #     if self.penalize_cnots:
    #         penalty = self.pool.get_cnots(index)
    #     else:
    #         penalty = 1
    #     gradient = gradient/penalty
    #     return gradient

            

    # def grow_and_update(self, viable_candidates, viable_gradients):
