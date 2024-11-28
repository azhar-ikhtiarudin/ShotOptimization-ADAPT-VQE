
import numpy as np

from scipy.sparse import csc_matrix
from copy import deepcopy

from src.utilities import get_hf_det, ket_to_vector, bfgs_update, to_qiskit_operator
from src.sparse_tools import get_sparse_operator 
from src.pools import ImplementationType
from src.minimize import minimize_bfgs
from openfermion import jordan_wigner

from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import AerSimulator

from .adapt_data import AdaptData

class AdaptVQE():

    def __init__(self,
                 pool,
                 molecule,
                 max_adapt_iter,
                 max_opt_iter,
                 verbose,
                 candidates=1,
                 threshold=0.1,
                 full_opt=True,
                 recycle_hessian=False,
                 orb_opt=False,
                 convergence_criterion="total_g_norm",
                 rand_degenerate = False
                 ):
        
        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.verbose = verbose
        self.candidates = candidates
        self.threshold = threshold
        self.full_opt = full_opt
        self.recycle_hessian = recycle_hessian
        self.orb_opt = orb_opt
        self.convergence_criterion = convergence_criterion
        self.rand_degenerate = rand_degenerate
        

        self.initialize_hamiltonian()
        self.file_name = (
            f"{self.molecule.description}_r={self.molecule.geometry[1][1][2]}"
        )
        self.gradients = np.array(())
        self.create_orb_rotation_ops()
        self.orb_opt_dim = len(self.orb_ops)
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
            self.qubit_hamiltonian = jordan_wigner(hamiltonian)
            self.exact_energy = self.molecule.fci_energy
        
        if self.verbose:
            print("HF State:", self.ref_det)
            print("N Qubits:", self.n)
            print("Hamiltonian Type:", type(self.molecule.get_molecular_hamiltonian()))
            print("Hamiltonian:", self.molecule.get_molecular_hamiltonian())
            print("Qubit Hamiltonian:", self.qubit_hamiltonian)
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
            state = self.pool.expm_mult(coefficient, index, state)
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
         print("Data", self.data)
        #  qc = self.pool.get_circuit(self.data.result.ansatz.indices, self.data.result.ansatz.coefficients)
        #  print(qc)

         ket = self.get_state(coefficients, indices, ref_state)

        #  if orb_params is not None:
        #     orb_rotation_generator = self.create_orb_rotation_generator(orb_params)
        #     ket = expm_multiply(orb_rotation_generator, ket)

         bra = ket.transpose().conj()
         exp_value = (bra.dot(observable.dot(ket)))[0,0].real

        #  ket = ket[:,0]
         print("\n- - - == Evaluate Observable == - - -")
         print("self.coefficients", coefficients)
         print("self.indices", indices)
         print("ket:", ket)
         print("ket type:", type(ket))
         print("bra:", bra)
         print("obs:\n", observable)
         print("res:", exp_value, '\n')

         return exp_value
    
    def evaluate_observable_sampler(self, 
                            observable,
                            coefficients=None,
                            indices=None,
                            ref_state=None,
                            orb_params=None
                            ):
        from qiskit import QuantumCircuit
        
        if coefficients is None and indices is None:
            print("Coefficients and Indices is None, initiate circuit")
            qc = QuantumCircuit(self.n)
        
        else:
            print("Coefficients and Indices is not None, get_circuit()")
            qc = self.pool.get_circuit(self.data.result.ansatz.indices, self.data.result.ansatz.coefficients)

        print("")
        X = Pauli("X")
        Z = Pauli("Z")
        I = Pauli("I")
        Y = Pauli("Y")
        sampler = SamplerV2(backend=AerSimulator())
        shots = 1000

        
        qc_updates = []
        energy = 0
        print("Type Observable:", type(observable))
        commuted_hamiltonian = observable.group_commuting(qubit_wise=True)

        for cliques in commuted_hamiltonian:
            # print(cliques)
            qc_update = qc.copy()
            i = 0
            for pauli in cliques[0].paulis[0]:
                # print(pauli, i)
                if (pauli == Y):
                    qc_update.sdg(i)
                    qc_update.h(i)
                elif (pauli == X):
                    # qc_update.sdg(i)
                    qc_update.h(i)
                i += 1
            qc_update.measure_all()
            qc_updates.append(qc_update) 

            # Circuit Obtained
            job = sampler.run([(qc_update)], shots=shots)
            counts = job.result()[0].data.meas.get_counts()
            # print(counts)

            probs = self.get_probability_distribution(counts, shots, self.n)
            # print(probs, "\n")


            for pauli_string in cliques:
                eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                # print(eigen_value)

                res = np.dot(eigen_value, probs)*pauli_string.coeffs
                energy += res
            
            #     print("Result:", res)
            #     print("Energy:", energy)
                
            #     print("\n")

            # print("\n")

        print(energy)
        return energy


    
    def get_probability_distribution(counts, NUM_SHOTS, N):
        # Generate all possible N-qubit measurement outcomes
        all_possible_outcomes = [''.join(format(i, '0' + str(N) + 'b')) for i in range(2**N)]
        
        # Ensure all possible outcomes are in counts
        for k in all_possible_outcomes:
            if k not in counts.keys():
                counts[k] = 0
        
        # Sort counts by outcome
        sorted_counts = sorted(counts.items())
        
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
        
        for pauli in reversed(pauli_strings):
            # Start with identity matrix
            # print(pauli)
            # print(pauli_dict[pauli])
            eigen_vals = np.kron(eigen_vals, pauli_dict[pauli])
            
            # Apply the corresponding Pauli matrix for each qubit
            # for char in pauli:
            #     op = np.kron(op, pauli_dict[char])
            
            # eigen_vals.append(op)
        
        return eigen_vals
        
    def run(self):
        # Run Full ADAPT-VQE Algorithm
        self.initialize()
        
        finished = False
        print("First Run, finished=", finished)
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()

        # End of the loop
        if not finished:
            viable_candidates, viable_gradients, total_norm, max_norm = (
                self.rank_gradients()
            )
            finished = self.probe_termination(total_norm, max_norm)

        if finished:
            print("Converged!")
            error = self.energy - self.exact_energy
        else:
            print("Maximum iteration reached before converged!")
            self.data.close(False)
    
    def run_iteration(self):
        # Run one Iteration of the algorithm
        print("=== Run Iteration ===")
        finished, viable_candidates, viable_gradients, total_norm = (
            self.start_iteration()
        )

        if finished:
            return finished # already converged
        
        while viable_candidates:
            print("WHILE VIABLE CANDIDATES:", viable_candidates, viable_gradients)
            energy, g, viable_candidates, viable_gradients = self.grow_and_update(
                viable_candidates, viable_gradients
            )
            print("WHILE VIABLE CANDIDATES:", energy, g, viable_candidates, viable_gradients)

        if energy is None:
            energy = self.optimize(g)
        
        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished
    
    def initialize(self):
        initial_energy = self.evaluate_observable(self.hamiltonian)
        # initial_energy = self.evaluate_energy_sampler()
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
            
            self.indices = []
            self.coefficients = []
            self.old_coefficients = []
            self.old_gradients = []

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
            # print("Operator Type", type(operator))
            # print("----Operator", operator)
            # operator_qiskit = to_qiskit_operator(operator, little_endian=False)
            # print("----Operator Converted", operator_qiskit)
            observable = 2 * self.hamiltonian @ operator

            # measurement = self.observable_to_measurement(observable)
            # self.pool.store_grad_meas(index, observable)
        
        gradient = self.evaluate_observable(observable, coefficients, indices)

        return gradient
    
    def place_gradient(self, gradient, index, sel_gradients, sel_indices):
        i = 0
        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8:
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
        print("gradient", gradient)
        print("sel_gradient", sel_gradient)
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


    def grow_and_update(self, viable_candidates, viable_gradients):
        
        energy, gradient = self.grow_ansatz(viable_candidates, viable_gradients)

        viable_candidates, viable_gradients, extra_ngevs = (
            self.update_viable_candidates(viable_candidates, viable_gradients)
        )

        self.iteration_sel_gradients = np.append(self.iteration_sel_gradients, gradient)
        return energy, gradient, viable_candidates, viable_gradients


    def grow_ansatz(self, viable_candidates, viable_gradients, max_additions=1):

        total_new_nfevs = []
        total_new_ngevs = []
        total_new_nits = []
        gradients = []

        while max_additions > 0:
            energy, gradient, new_nfevs, new_ngevs, new_nits = self.select_operators(
                viable_candidates, viable_gradients
            )
            if self.data.evolution.indices:
                old_size = len(self.data.evolution.indices[-1])
            else:
                old_size = 0
            new_indices = self.indices[old_size:]

            if new_nfevs:
                total_new_nfevs.append(new_nfevs)
            if new_ngevs:
                total_new_ngevs.append(new_nits)
            if new_nits:
                total_new_nits.append(new_nits)
            
            gradients.append(gradient)
            max_additions -= 1
        
        print("Operator(s) added to ansatz:", new_indices)
        self.update_iteration_costs(total_new_nfevs, total_new_ngevs, total_new_nits)

        return energy, gradient

    def select_operators(self, max_indices, max_gradients):
        
        new_nfevs = []
        new_ngevs = []
        new_nits = []
        energy = None

        gradient = self.select_via_gradient(max_indices, max_gradients)

        return energy, gradient, new_nfevs, new_ngevs, new_nits
    
    def select_via_gradient(self, indices, gradients):

        index, gradient = self.find_highest_gradient(indices, gradients)

        # Grow the ansatz and the parameter and gradient vectors
        self.indices.append(index)
        self.coefficients.append(0)
        self.gradients = np.append(self.gradients, gradient)

        return gradient
    
    def find_highest_gradient(self, indices, gradients, excluded_range=[]):

        viable_indices = []
        viable_gradients = []
        for index, gradient in zip(indices, gradients):
            if index not in excluded_range:
                viable_indices.append(index)
                viable_gradients.append(gradient)
        
        abs_gradients = [ np.abs(gradient) for gradient in viable_gradients ]
        max_abs_gradient = max(abs_gradients)

        grad_rank = abs_gradients.index(max_abs_gradient)
        index = viable_indices[grad_rank]
        gradient = viable_gradients[grad_rank]

        return index, gradient
    
    def update_iteration_costs(self, new_nfevs=None, new_ngevs=None, new_nits=None):
        if new_nfevs:
            self.iteration_nfevs = self.iteration_nfevs + new_nfevs
        if new_ngevs:
            self.iteration_ngevs = self.iteration_ngevs + new_ngevs
        if new_nits:
            self.iteration_nits = self.iteration_nits + new_nits

    def update_viable_candidates(self, viable_candidates, viable_gradients):
        viable_candidates = []
        ngevs = 0
        return viable_candidates, viable_gradients, ngevs
    


        
    def optimize(self, gradient):
        if not self.full_opt:
            energy, nfev, g1ev, nit = self.partial_optim(gradient)
        else:
            self.inv_hessian = self.expand_inv_hessian()

            (
                self.coefficients,
                energy,
                self.inv_hessian,
                self.gradients,
                nfev,
                g1ev,
                nit
            ) = self.full_optim()
        

        print("=After Optimization=")
        print("self.coefficients", self.coefficients)
        print("energy", energy)
        print("gradient", gradient)

        self.iteration_nfevs.append(nfev)
        self.iteration_ngevs.append(g1ev)
        self.iteration_nits.append(nit)
        return energy
    


    
    def full_optim(
            self,
            indices=None,
            initial_coefficients=None,
            initial_inv_hessian=None,
            e0=None,
            g0=None,
            maxiters=None
    ):
        print(". . . == Full Optimization == . . .")
        initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters = (
            self.prepare_opt(
                initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters
            )
        )

        print(
            f"Initial Energy: {self.energy}"
            f"\nOptimizing energy with indices {list(indices)}..."
            f"\nStarting point: {list(initial_coefficients)}"
        )

        evolution = {
            "parameters":[],
            "energy":[],
            "inv_hessian":[],
            "gradient":[]
        }

        def callback(args):
            evolution['parameters'].append(args.x)
            evolution['energy'].append(args.fun)
            evolution['inv_hessian'].append(args.inv_hessian)
            evolution['gradient'].append(args.gradient)
        
        # define cost function
        e_fun = lambda x, ixs: self.evaluate_energy(
            coefficients=x[self.orb_opt_dim:],
            indices=ixs,
            orb_params=x[:self.orb_opt_dim]
        )

        extra_njev = 0

        if self.recycle_hessian and (not self.data.iteration_counter and self.orb_opt):
            g0 = self.estimate_gradients(initial_coefficients, indices)
            extra_njev = 1
        
        opt_result = minimize_bfgs(
            e_fun,
            initial_coefficients,
            [indices],
            jac=self.estimate_gradients,
            initial_inv_hessian=initial_inv_hessian,
            disp=self.verbose,
            gtol=10**-8,
            maxiter = self.max_opt_iter,
            callback= callback,
            f0=e0,
            g0=g0
        )

        opt_coefficients = list(opt_result.x)
        orb_params = opt_coefficients[:self.orb_opt_dim]
        ansatz_coefficients = opt_coefficients[self.orb_opt_dim:]
        opt_energy = self.evaluate_observable(self.hamiltonian, ansatz_coefficients, indices, ref_state=None,orb_params=orb_params)

        # self.perform_sim_transform(orb_params)

        # Add costs
        nfev = opt_result.nfev
        njev = opt_result.njev + extra_njev
        ngev = njev * len(indices)
        nit = opt_result.nit

        if self.recycle_hessian:
            inv_hessian = opt_result.hess_inv
        else:
            inv_hessian = None

        if opt_result.nit:
            gradients = evolution["gradient"][-1]
        else:
            gradients = g0

        return ansatz_coefficients, opt_energy, inv_hessian, gradients, nfev, ngev, nit

    def prepare_opt(
        self, initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters
    ):
        """
        Prepares the arguments for the optimization by replacing None arguments with defaults.

        Args:
            initial_coefficients (list): the initial point for the optimization. If None, the initial point will be the
                previous coefficients with zeros appended.
            indices (list): the indices defining the ansatz before the new addition. If None, current ansatz is assumed
            initial_inv_hessian (np.ndarray): an approximation for the initial inverse Hessian for the optimization
            e0 (float): initial energy
            g0 (list): initial gradient vector
            maxiters (int): maximum number of optimizer iterations. If None, self.max_opt_iters is assumed.
        """

        if initial_coefficients is None:
            initial_coefficients = deepcopy(self.coefficients)
        if indices is None:
            indices = self.indices.copy()
        if initial_coefficients is None and indices is None:
            # Use current Hessian, gradient and energy for the starting point
            if initial_inv_hessian is None:
                initial_inv_hessian = self.inv_hessian
            if g0 is None and self.recycle_hessian:
                g0 = self.gradients
            if e0 is None and self.recycle_hessian:
                e0 = self.energy
        if maxiters is None:
            maxiters = self.max_opt_iter
            
        initial_coefficients = np.append(
            [0 for _ in range(self.orb_opt_dim)], initial_coefficients
        )

        return initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters
    
        


    def bfgs_update(self, gfkp1, gfk, xkp1, xk):
        """
        Perform a BFGS update on the inverse Hessian matrix.

        Arguments:
            gfkp1 (Union[list,np.ndarray]): the gradient at the new iterate
            gfk (Union[list,np.ndarray]): the gradient at the old iterate
            xkp1 (nion[list,np.ndarray]): the coefficients at the new iterate
            xk (Union[list,np.ndarray]): the coefficients at the old iterate

        Returns:
            inv_hessian (np.ndarray): the updated hessian. Also updates self.inv_hessian
        """

        gfkp1 = np.array(gfkp1)
        gfk = np.array(gfk)
        xkp1 = np.array(xkp1)
        xk = np.array(xk)

        self.inv_hessian = bfgs_update(self.inv_hessian, gfkp1, gfk, xkp1, xk)

        return self.inv_hessian
    

    def expand_inv_hessian(self, added_dim=None):
        """
        Expand the current approximation to the inverse Hessian by adding ones in the diagonal, zero elsewhere.

        Arguments:
            added_dim (int): the number of added dimensions (equal to the number of added lines/columns). If None,
                it is assumed that the Hessian is expanded so that its dimension is consistent with the current ansatz.

        Returns:
            inv_hessian (np.ndarray): the expanded inverse Hessian
        """

        if not self.recycle_hessian:
            return None

        size, size = self.inv_hessian.shape

        if added_dim is None:
            # Expand Hessian to have as many columns as the number of ansatz + orbital optimization parameters
            added_dim = len(self.indices) + self.orb_opt_dim - size

        size += added_dim

        # Expand inverse Hessian with zeros
        inv_hessian = np.zeros((size, size))
        inv_hessian[:-added_dim, :-added_dim] += self.inv_hessian

        # Add ones in the diagonal
        while added_dim > 0:
            inv_hessian[-added_dim, -added_dim] = 1
            added_dim -= 1

        return inv_hessian
    
    def create_orb_rotation_ops(self):
        """
        Create list of orbital rotation operators.
        See https://doi.org/10.48550/arXiv.2212.11405
        """

        n_spatial = int(self.n / 2)

        k = 0
        self.orb_ops = []
        self.sparse_orb_ops = []

        if not self.orb_opt:
            return

        for p in range(n_spatial):
            for q in range(p + 1, n_spatial):
                new_op = create_spin_adapted_one_body_op(p, q)
                # new_op = get_sparse_operator(new_op, n_spatial * 2)
                self.orb_ops.append(new_op)
                sparse_op = get_sparse_operator(new_op, self.n)
                self.sparse_orb_ops.append(sparse_op)
                k += 1

        assert len(self.orb_ops) == int((n_spatial * (n_spatial - 1)) / 2)

        return
    
    def estimate_gradients(
        self, coefficients=None, indices=None, method="an", dx=10**-8, orb_params=None
    ):
        """
        Estimates the gradients of all operators in the ansatz defined by coefficients and indices. If they are None,
        the current state is assumed. Default method is analytical (with unitary recycling for faster execution).

        Args:
            coefficients (list): the coefficients of the ansatz. If None, current coefficients will be used.
            indices (list): the indices of the ansatz. If None, current indices will be used.
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradients (list): the gradient vector
        """

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().estimate_gradients(
                coefficients=coefficients, indices=indices, method=method, dx=dx
            )

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if indices is None:
            assert coefficients is None
            indices = self.indices
            coefficients = self.coefficients

        if self.orb_opt:
            orb_params = coefficients[: self.orb_opt_dim]
            coefficients = coefficients[self.orb_opt_dim :]
        else:
            orb_params = None

        if not len(indices):
            return []

        # Define orbital rotation
        hamiltonian = self.hamiltonian
        if orb_params is not None:
            generator = self.create_orb_rotation_generator(orb_params)
            orb_rotation = expm(generator)
            hamiltonian = (
                orb_rotation.transpose().conj().dot(hamiltonian).dot(orb_rotation)
            )
        else:
            orb_rotation = np.eye(2**self.n)
            orb_rotation = csc_matrix(orb_rotation)

        gradients = []
        state = self.compute_state(coefficients, indices)
        right_matrix = self.sparse_ref_state
        left_matrix = self.compute_state(
            coefficients, indices, hamiltonian.dot(state), bra=True
        )

        # Ansatz gradients
        for operator_pos in range(len(indices)):
            operator = self.pool.get_imp_op(indices[operator_pos])
            coefficient = coefficients[operator_pos]
            index = indices[operator_pos]

            left_matrix = (
                self.pool.expm_mult(coefficient, index, left_matrix.transpose().conj())
                .transpose()
                .conj()
            )
            right_matrix = self.pool.expm_mult(coefficient, index, right_matrix)

            gradient = 2 * (left_matrix.dot(operator.dot(right_matrix)))[0, 0].real
            gradients.append(gradient)

        right_matrix = csc_matrix(orb_rotation.dot(right_matrix))
        left_matrix = csc_matrix(right_matrix.transpose().conj())

        # Orbital gradients
        orb_gradients = []
        for operator in self.sparse_orb_ops:
            gradient = (
                2
                * left_matrix.dot(self.hamiltonian)
                .dot(operator)
                .dot(right_matrix)[0, 0]
                .real
            )
            orb_gradients.append(gradient)

        # Remember that orbital optimization coefficients come first
        gradients = orb_gradients + gradients

        return gradients

    def evaluate_energy(
        self, coefficients=None, indices=None, ref_state=None, orb_params=None
    ):
        """
        Calculates the energy in a specified state using matrix algebra.
        If coefficients and indices are not specified, the current ones are used.

        Arguments:
          coefficients (list): coefficients of the ansatz
          indices (list): indices of the ansatz
          ref_state (csc_matrix): the reference state to which to append the ansatz
          orb_params (list): if self.orb_params, the parameters of the orbital rotation operators

        Returns:
          energy (float): the energy in this state.
        """
        print("Evaluate Observable")
        energy = self.evaluate_observable(
            self.hamiltonian, coefficients, indices, ref_state, orb_params
        )

        return energy
    
    def evaluate_energy_sampler(self):
        print(self.data)
        print(self.coefficients)
        print(self.indices)


    def complete_iteration(self, energy, total_norm, sel_gradients):
        """
        Complete iteration by storing the relevant data and updating the state.

        Arguments:
            energy (float): final energy for this iteration
            total_norm (float): final gradient norm for this iteration
            sel_gradients(list): gradients selected in this iteration
        """

        energy_change = energy - self.energy
        self.energy = energy

        # Save iteration data
        self.data.process_iteration(
            self.indices,
            self.energy,
            total_norm,
            sel_gradients,
            self.coefficients,
            self.inv_hessian,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
        )

        # Update current state
        self.state = self.compute_state()

        print("\nCurrent energy:", self.energy)
        print(f"(change of {energy_change})")
        print(f"Current ansatz: {list(self.indices)}")

        return
    
    def converged(self):
        """
        To call when convergence is reached. Updates file name to include the total number of iterations executed.
        """

        # Update iteration number on file_name - we didn't reach the maximum
        # pre_it, post_it = self.file_name.split(str(self.max_adapt_iter) + "i")
        # self.file_name = "".join(
        #     [pre_it, str(self.data.iteration_counter) + "i", post_it]
        # )
        self.data.close(True, self.file_name)

        return