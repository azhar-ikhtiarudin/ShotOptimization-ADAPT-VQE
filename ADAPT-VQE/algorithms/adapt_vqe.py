
import numpy as np

from scipy.sparse import csc_matrix
from copy import deepcopy

from src.utilities import get_hf_det, ket_to_vector
from src.sparse_tools import get_sparse_operator

from .adapt_data import AdaptData

class AdaptVQE():

    def __init__(self,
                 pool,
                 molecule,
                 max_adapt_iter,
                 max_opt_iter,
                 verbose
                 ):
        
        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.verbose = verbose

        self.initialize_hamiltonian()
        self.file_name = (
            f"{self.molecule.description}_r={self.molecule.geometry[1][1][2]}"
        )
        self.gradients = np.array(())
        self.data = None

        # Matrix based
        self.state = self.sparse_ref_state
        self.ref_state = self.sparse_ref_state
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
         
         print("Evaluate observable, ket:", ket)
         print("Evaluate observable, bra:", bra)
         
         exp_value = (bra.dot(observable.dot(ket)))[0,0].real
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
    
    def evaluate_energy(self):
        pass