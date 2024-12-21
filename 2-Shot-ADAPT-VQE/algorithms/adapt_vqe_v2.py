import time
from copy import copy, deepcopy
import numpy as np
from scipy.sparse import csc_matrix

from .adapt_data import AdaptData
from src.minimize import minimize_bfgs
from src.utilities import ket_to_vector
from src.circuits import pauli_exp_circuit

from openfermion.transforms import jordan_wigner
from openfermion.utils import commutator
from openfermion.linalg import get_sparse_operator

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import EstimatorV2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.primitives import StatevectorEstimator

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from scipy.optimize import minimize
from src.utilities import to_qiskit_operator


class ImplementationType:
    SPARSE = 0
    QISKIT = 1

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
        self.qubit_hamiltonian_sparse = get_sparse_operator(self.qubit_hamiltonian, self.n)

        self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)
        
        self.exact_energy = self.molecule.fci_energy
        
        self.window = self.pool.size

        # Hartree Fock Reference State:

        self.ref_determinant = [ 1 for _ in range(self.molecule.n_electrons) ]
        self.ref_determinant += [ 0 for _ in range(self.fermionic_hamiltonian.n_qubits - self.molecule.n_electrons ) ]

        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype=complex
        ).transpose()


        self.ref_circuit = QuantumCircuit(self.n)
        # Reference State Circuit
        
        for i, qubit in enumerate(self.ref_determinant):
            if qubit == 1 : 
                self.ref_circuit.x(i)
        self.ref_circuit.barrier()
        # self.reference_circuit = qc
        # self.qc_optimized = qc

        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype = complex
            ).transpose()
        
        self.gradients = np.array(())

        # if self.vrb:
            # print("\n. . . ========== ADAPT-VQE Settings ========== . . .")
            # print("\nFermionic Hamiltonian:", self.fermionic_hamiltonian)
            # print("\nQubit Hamiltonian:", self.qubit_hamiltonian)
            # print("\nHartree Fock Reference State:", self.ref_determinant)
            # print("\nHartree Fock Reference State Circuit:", qc)


    def run(self):
        if self.vrb: print("\n. . . ======= Start Run ADAPT-VQE ======= . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()
        
        if not finished:
            viable_candidates, viable_gradients, total_norm, max_norm = (self.rank_gradients())
            if total_norm < 10**-8:
                self.data.close(True) # converge()
                finished = True
            print("Self.Energy", self.energy)
            if np.abs(self.energy - self.exact_energy) <= 10**-4:
                finished = True
        
        if finished:
            print("\n. . . ======= Convergence Condition Achieved 🎉🎉🎉 ======= . . .")
            error = self.energy - self.exact_energy
            print(f"\n\t> Energy:")
            print(f"\tFinal Energy = {self.energy}")
            print(f"\tError = {error}")
            print(f"\tError in Chemical accuracy= {error/(1.5936 * 10**-3)*100}")
            print(f"\tIterations completed = {self.data.iteration_counter}")

            print(f"\n\t> Circuit Property:")
            print(f"\tAnsatz indices = {self.indices}")
            print(f"\tCoefficients = {self.coefficients}")

        else:
            print("\n. . . ======= Maximum iteration reached before converged! ======= . . . \n")
            self.data.close(False)
        
        return
            


    def initialize(self):
        if self.vrb:
            print("\n # Initialize Data ")
        if not self.data:
            self.indices = []
            self.coefficients = []
            self.old_coefficients = []
            self.old_gradients = []

        # self.initial_energy = self.evaluate_observable(self.qubit_hamiltonian, disp=True) 
        self.initial_energy = self.evaluate_energy()
        self.energy = self.initial_energy
        print("\tInitial Energy = ", self.initial_energy)
        print('\tExact Energt =', self.exact_energy)

        if not self.data: 
            self.data = AdaptData(self.initial_energy, self.pool, self.exact_energy, self.n)
        return


    def run_iteration(self):

        # Gradient Screening
        finished, viable_candidates, viable_gradients, total_norm = ( 
            self.start_iteration() 
        )
        
        if finished: 
            return finished

        while viable_candidates:
            energy, gradient, viable_candidates, viable_gradients = self.grow_and_update( 
                viable_candidates, viable_gradients 
            )
            
        if energy is None: 
            energy = self.optimize(gradient) # Optimize energy with current updated ansatz (additional gradient g)

        if np.abs(energy - self.exact_energy) <= 10**-3:
            finished = True

        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished

    
    def start_iteration(self):
        
        if self.vrb: print(f"\n. . . ======= ADAPT-VQE Iteration {self.data.iteration_counter + 1} ======= . . .")
        
        print(f"\n # Active Circuit at Adapt iteration {self.data.iteration_counter + 1}:")
        # print(self.qc_optimized)
        
        viable_candidates, viable_gradients, total_norm, max_norm = ( 
            self.rank_gradients() 
        )

        finished = False
        if total_norm < self.grad_threshold:
            self.data.close(True) # converge()
            finished = True
        
        if finished: return finished, viable_candidates, viable_gradients, total_norm

        print(
            # f"\n\tViable Operator Candidates: {viable_candidates}"
            # f"\n\tViable Operator Gradients: {viable_gradients}"
            f"\tIs Finished? -> {finished}"
        )

        self.iteration_nfevs = []
        self.iteration_ngevs = []
        self.iteration_nits = []
        self.iteration_sel_gradients = []
        self.iteration_qubits = ( set() )

        return finished, viable_candidates, viable_gradients, total_norm
        


    def rank_gradients(self, coefficients=None, indices=None):
        
        print(f"\n # Rank Gradients (Pool size = {self.pool.size})")

        sel_gradients = []
        sel_indices = []
        total_norm = 0

        # if self.vrb: print("Total Norm Initial:", total_norm)

        for index in range(self.pool.size):

            if self.vrb: print("\n\tEvaluating Gradient", index)

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            
            if self.vrb: print(f"\t\tvalue = {gradient}")

            if np.abs(gradient) < 10**-8:
                continue

            sel_gradients, sel_indices = self.place_gradient( 
                gradient, index, sel_gradients, sel_indices 
            )

            if index not in self.pool.parent_range: 
                total_norm += gradient**2
                print(f"\t\ttotal norm = {total_norm} ✅")

        total_norm = np.sqrt(total_norm)

        if sel_gradients: 
            max_norm = sel_gradients[0]
        else: 
            max_norm = 0

        if self.vrb:
            print("\n # Gradient Rank Total Results")
            print(f"\n\tTotal gradient norm: {total_norm}")
            print("\tFinal Selected Indices:", sel_indices)
            print("\tFinal Selected Gradients:", sel_gradients)
            # print("\t\tTotal Norm", total_norm)
            # print("\t\tMax Norm", max_norm)
        
        return sel_indices, sel_gradients, total_norm, max_norm
    
    
    # def eval_candidate_gradient_qiskit(self, index, coefficients=None, indices=None):
    #     observable = self.pool.get_grad_meas(index)
        
    #     if observable is None:
    #         operator = self.pool.get_q_op(index)
    #         print("Gradient Measurement")
    #         print("Hamiltonian:", self.qubit_hamiltonian)
    #         print("Operator OpenFermion:", operator)
    #         operator_sparse = get_sparse_operator(operator)
    #         print("Operator Sparse:", operator_sparse)
            

    #         observable = commutator(self.qubit_hamiltonian, operator)

            
    #         self.pool.store_grad_meas(index, observable)
        
    #     gradient = self.evaluate_observable(observable, coefficients, indices)

    #     return gradient
    

    
    def eval_candidate_gradient(self, index, coefficients=None, indices=None):

        import time

        # observable = self.pool.get_grad_meas(index)
        # t0 = time.time()
        self.pool.imp_type = ImplementationType.SPARSE

        # if observable is None:
        operator = self.pool.get_q_op(index)


        operator_sparse = get_sparse_operator(operator, self.n)

        t0 = time.time()
        observable_sparse = 2 * self.qubit_hamiltonian_sparse @ operator_sparse
        # print("\nSparse Observable:", observable_sparse)

        # observable = commutator(self.qubit_hamiltonian, operator)

        # self.pool.store_grad_meas(index, observable)

        ket = self.get_state(self.coefficients, self.indices, self.sparse_ref_state)
        
        bra = ket.transpose().conj()

        gradient = (bra.dot(observable_sparse.dot(ket)))[0,0].real
        tf = time.time()
        print(f"\t\tEstimator Time = {tf-t0}")
        

        return gradient

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        state = self.sparse_ref_state
        if coefficients is None or indices is None:
            return state
        else:
            for coefficient, index in zip(coefficients, indices):
                state = self.pool.expm_mult(coefficient, index, state)
        return state


    # def evaluate_observable(self, observable, disp=False, coefficients=None, indices=None):

    #     qiskit_observable = to_qiskit_operator(observable)

    #     qc = self.qc_optimized
        
    #     estimator = StatevectorEstimator()
    #     # pass_manager = generate_preset_pass_manager(3, AerSimulator())
    #     # isa_circuit = pass_manager.run(qc)
    #     pub = (qc, qiskit_observable)
    #     job = estimator.run([pub])
    #     exp_vals = job.result()[0].data.evs

    #     if disp == True:
    #         print("\n/start evaluate observable functions/")
    #         print("> coefficients", self.coefficients)
    #         print("> observables", qiskit_observable)
    #         print("evaluated circuit:", qc)
    #         print("expectation values =", exp_vals)
    #         print("\n/end evaluate observable functions/")

    #     return exp_vals
    
    def place_gradient(self, gradient, index, sel_gradients, sel_indices):

        i = 0

        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8:
                condition = self.break_gradient_tie(gradient, sel_gradient)
                if condition: break
            
            elif np.abs(gradient) - np.abs(sel_gradient) >= 10**-8:
                break

            i += 1
        
        # print("Self.Window:", self.window)
        if i < self.window:
            sel_indices = sel_indices[:i] + [index] + sel_indices[i : self.window - 1]

            sel_gradients = (
                sel_gradients[:i] + [gradient] + sel_gradients[i : self.window - 1]
            )
        
        return sel_gradients, sel_indices

    def break_gradient_tie(self, gradient, sel_gradient):
        assert np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8

        condition = np.abs(gradient) > np.abs(sel_gradient)

        return condition
    

    def grow_and_update(self, viable_candidates, viable_gradients):
        print("\n # Grow and Update Ansatz")
        
        # Grow Ansatz
        energy, gradient = self.grow_ansatz(viable_candidates, viable_gradients)

        # Update Viable Candidates
        viable_candidates = []

        self.iteration_sel_gradients = np.append(self.iteration_sel_gradients, gradient)
        return energy, gradient, viable_candidates, viable_gradients


    def grow_ansatz(self, viable_candidates, viable_gradients, max_additions=1):

        total_new_nfevs = []
        total_new_ngevs = []
        total_new_nits = []
        gradients = []
 
        while max_additions > 0:

            new_nfevs = []
            new_ngevs = []
            new_nits = []
            energy = None

            index, gradient = self.find_highest_gradient(viable_candidates, viable_gradients)

            # Grow the ansatz and the parameter and gradient vectors
            print("\tGrow ansatz with parameter coefficients:", self.coefficients)
            
            # np.append(self.indices, index)
            print("\t\tself.coefficients", self.coefficients)
            print("\t\tself.indices", self.indices)
            print("\t\tself.coefficients", type(self.coefficients))
            print("\t\tself.indices", type(self.indices))

            self.indices.append(index)
            # self.coefficients.append(0)
            self.coefficients = np.append(self.coefficients, 0)

            print("\t\tself.coefficients updated", self.coefficients)
            print("\t\tself.indices updated", self.indices)

            # np.append(self.coefficients, 0)

            self.gradients = np.append(self.gradients, gradient)

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
        
        print("\tOperator(s) added to ansatz:", new_indices)
        self.update_iteration_costs(total_new_nfevs, total_new_ngevs, total_new_nits)

        return energy, gradient

    
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
            # self.inv_hessian,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
        )

        # Update current state
        # self.state = self.compute_state()
        print("\n # Complete Iteration")
        print("\tCurrent energy:", self.energy, "change of", energy_change)
        print(f"\tCurrent ansatz: {list(self.indices)}")

        return
    
    def optimize(self, gradient):
        """gradient: gradient of the last-added operator"""

        # Full Optimization
        print("\n # Standard VQE Optimization")

        
        initial_coefficients = deepcopy(self.coefficients)
        indices = self.indices.copy()
        g0 = self.gradients
        e0 = self.energy
        maxiters = self.max_opt_iter

        print("\n\tEnergy Optimization Parameter")
        print("\t\tInitial Coefficients:", initial_coefficients)
        print("\t\tIndices:", indices)
        print("\t\tg0:", g0)
        print("\t\te0:", e0, "\n")

        parameters = ParameterVector("theta", len(indices))
        qc = self.pool.get_circuit(indices, initial_coefficients, parameters)
        # print("\nQC", qc)
        # print("\nself.reference_circuit", self.reference_circuit)

        ansatz = self.ref_circuit.compose(qc)

        print("\tAnsatz Circuit:\n", ansatz)

        print(
            f"\nOptimization Property"
            f"\n\tInitial energy: {self.energy}"
            f"\n\tExact energy: {self.exact_energy}"
            f"\n\tError: {self.exact_energy-self.energy}"
            f"\n\tOptimizing energy with indices {list(indices)}..."
            f"\n\tStarting point: {list(initial_coefficients)}"
            f"\n\tNumber of Parameters: {ansatz.num_parameters}"
            f"\n\nIterations:"
        )

        cost_history_dict = {
            "prev_vector": None,
            "iters":0,
            "cost_history":[]
        }

        estimator = Estimator()

        def cost_function(params, ansatz, hamiltonian, estimator):
            """Return estimate of energy from estimator"""

            pass_manager = generate_preset_pass_manager(3, AerSimulator())
            isa_ansatz = pass_manager.run(ansatz)

            pub = (isa_ansatz, [hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]

            cost_history_dict['iters'] += 1
            cost_history_dict['previous_vector'] = params
            cost_history_dict['cost_history'].append(energy)

            print("\t", cost_history_dict['iters'], "\tE =", energy)
            return energy

        res = minimize(
            cost_function,
            initial_coefficients,
            args=(ansatz, self.qiskit_hamiltonian, estimator),
            method='bfgs',
            
        )

        print("\nScipy Optimize Result:",res)
        energy = cost_history_dict['cost_history'][-1]

        print("\nCoefficients and Indices")
        print(f"\n\tError Percentage: {(self.exact_energy - energy)/self.exact_energy*100}")
        print("\tself.coefficients initial:", self.coefficients)
        print("\tself.indices:", self.indices)
        
        self.coefficients = res.x
        print("\tself.coefficients updated:", self.coefficients)

        print("\nOptimized Circuit with Coefficients")
        # qc = self.pool.get_circuit_unparameterized(self.indices, self.coefficients)
        # self.qc_optimized = self.reference_circuit.compose(qc)
        # print(self.qc_optimized)

        return cost_history_dict['cost_history'][-1]
    
    
    def evaluate_energy(self, coefficients=None, indices=None):
        # energy = self.evaluate_observable(self.qubit_hamiltonian, coefficients, indices)
        # print("After Evaluate Energy:", energy)


        ## Qiskit Estimator
        # estimator = Estimator()
        estimator = StatevectorEstimator()
        self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)

        if indices is None or coefficients is None: 
            indices = []
            ansatz = self.ref_circuit
            pub = (ansatz, [self.qiskit_hamiltonian])

        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz = self.ref_circuit.compose(ansatz)
            pub = (ansatz, [self.qiskit_hamiltonian], [coefficients])


        result = estimator.run(pubs=[pub]).result()
        # print("Hamiltonian:", self.qiskit_hamiltonian)
        # print("Ansatz:", ansatz)
              
        energy_qiskit_estimator = result[0].data.evs[0]
        print("\t> Qiskit Estimator Energy Evaluation")
        print("\tenergy_qiskit_estimator:", energy_qiskit_estimator)

        return energy_qiskit_estimator
    
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
    