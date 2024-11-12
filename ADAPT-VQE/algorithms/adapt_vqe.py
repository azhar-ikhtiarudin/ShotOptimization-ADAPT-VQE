
from scipy.sparse import csc_matrix

from src.utilities import get_hf_det, ket_to_vector
from src.sparse_tools import get_sparse_operator

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
        pass

    def print_settings(self):
        pass

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        if coefficients is None and indices is None:
            state = self.state

        else:
            state = self.compute_state(coefficients, indices, ref_state)
        
        return state
    
    def run(self):
        self.initialize()
    
    def run_iterationa(self):
        pass