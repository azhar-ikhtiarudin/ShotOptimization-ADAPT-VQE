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
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qiskit_algorithms.optimizers import ADAM, SPSA

from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import Pauli
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

    def __init__(self, pool, molecule, max_adapt_iter, max_opt_iter, 
                 grad_threshold=10**-8, vrb=False, 
                 optimizer_method='bfgs', shots_assignment='vmsa',k=None, shots_budget=10000):

        self.pool = pool
        self.molecule = molecule
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.vrb = vrb
        self.grad_threshold = grad_threshold
        self.data = None
        self.n = self.molecule.n_qubits
        self.optimizer_method = optimizer_method
        self.shots_assignment = shots_assignment

        self.fermionic_hamiltonian = self.molecule.get_molecular_hamiltonian()
        self.qubit_hamiltonian = jordan_wigner(self.fermionic_hamiltonian)
        self.qubit_hamiltonian_sparse = get_sparse_operator(self.qubit_hamiltonian, self.n)
        self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)
        self.commuted_hamiltonian = self.qiskit_hamiltonian.group_commuting(qubit_wise=True)

        
        self.exact_energy = self.molecule.fci_energy
        self.window = self.pool.size

        self.k = k
        self.shots_budget = shots_budget
        self.shots_chemac = 0


        ## Qiskit
        # self.sampler = StatevectorSampler(seed=100)
        self.sampler = Sampler(seed=100)
        self.estimator = Estimator()
        self.PauliX = Pauli("X")
        self.PauliZ = Pauli("Z")
        self.PauliI = Pauli("I")
        self.PauliY = Pauli("Y")


        # Hartree Fock Reference State:
        self.ref_determinant = [ 1 for _ in range(self.molecule.n_electrons) ]
        self.ref_determinant += [ 0 for _ in range(self.fermionic_hamiltonian.n_qubits - self.molecule.n_electrons ) ]
        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype=complex
        ).transpose()


        # Reference State Circuit
        self.ref_circuit = QuantumCircuit(self.n)
        for i, qubit in enumerate(self.ref_determinant):
            if qubit == 1 : 
                self.ref_circuit.x(i)
        self.ref_circuit.barrier()

        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_determinant), dtype = complex
            ).transpose()
        

        self.cost_history_dict = {
            "prev_vector": None,
            "iters":0,
            "cost_history":[],
            'shots':[]
        }

        self.gradients = np.array(())
        self.iteration_nfevs = []
        self.iteration_ngevs = []
        self.iteration_nits = []
        self.total_norm = 0
        self.sel_gradients = []

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
            if total_norm < self.grad_threshold:
                self.data.close(True) # converge()
                finished = True
            print("Self.Energy", self.energy)
        
        if finished:
            print("\n. . . ======= Convergence Condition Achieved 🎉🎉🎉 ======= . . .")
            error = self.energy - self.exact_energy
            self.data.shots_chemac = self.shots_chemac
            print(f"\n\t> Energy:")
            print(f"\tFinal Energy = {self.energy}")
            print(f"\tError = {error}")
            print(f"\tError in Chemical accuracy= {error*627.5094} kcal/mol")
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

        self.initial_energy = self.evaluate_energy()
        self.energy = self.initial_energy
        print("\n\tInitial Energy = ", self.initial_energy)
        print('\tExact Energt =', self.exact_energy)

        self.energy_opt_iters = self.cost_history_dict['cost_history']
        self.shots_iters = self.cost_history_dict['shots']

        if not self.data: 
            self.data = AdaptData(self.initial_energy, self.pool, self.exact_energy, self.n)
        
        self.data.process_initial_iteration(
            self.indices,
            self.energy,
            self.total_norm,
            self.sel_gradients,
            self.coefficients,
            # 0,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
            self.energy_opt_iters,
            self.shots_iters
        )

        return


    def run_iteration(self):

        # Gradient Screening
        finished, viable_candidates, viable_gradients, total_norm = ( 
            self.start_iteration() 
        )

        self.energy_opt_iters = []
        self.shots_iters = []
        
        if finished: 
            return finished

        while viable_candidates:
            energy, gradient, viable_candidates, viable_gradients = self.grow_and_update( 
                viable_candidates, viable_gradients 
            )
            
        if energy is None: 
            energy = self.optimize(gradient) # Optimize energy with current updated ansatz (additional gradient g)

        self.complete_iteration(energy, total_norm, self.iteration_sel_gradients)

        return finished

    
    def start_iteration(self):
        
        if self.vrb: print(f"\n. . . ======= ADAPT-VQE Iteration {self.data.iteration_counter + 1} ======= . . .")
        
        # print(f"\n # Active Circuit at Adapt iteration {self.data.iteration_counter + 1}:")
        
        viable_candidates, viable_gradients, total_norm, max_norm = ( 
            self.rank_gradients() 
        )

        finished = False
        if total_norm < self.grad_threshold:
            self.data.close(True) # converge()
            finished = True
        
        if finished: 
            return finished, viable_candidates, viable_gradients, total_norm

        print(
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

        return sel_indices, sel_gradients, total_norm, max_norm
    
    
    def eval_candidate_gradient(self, index, coefficients=None, indices=None):

        self.pool.imp_type = ImplementationType.SPARSE

        operator = self.pool.get_q_op(index)
        operator_sparse = get_sparse_operator(operator, self.n)

        observable_sparse = 2 * self.qubit_hamiltonian_sparse @ operator_sparse

        ket = self.get_state(self.coefficients, self.indices, self.sparse_ref_state)        
        bra = ket.transpose().conj()
        gradient = (bra.dot(observable_sparse.dot(ket)))[0,0].real
        
        return gradient

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        state = self.sparse_ref_state
        if coefficients is None or indices is None:
            return state
        else:
            for coefficient, index in zip(coefficients, indices):
                state = self.pool.expm_mult(coefficient, index, state)
        return state

    def place_gradient(self, gradient, index, sel_gradients, sel_indices):

        i = 0

        for sel_gradient in sel_gradients:
            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8:
                condition = self.break_gradient_tie(gradient, sel_gradient)
                if condition: break
            
            elif np.abs(gradient) - np.abs(sel_gradient) >= 10**-8:
                break

            i += 1
        
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
            # print("\tGrow ansatz with parameter coefficients:", self.coefficients)
            
            # np.append(self.indices, index)
            # print("\t\tself.coefficients", self.coefficients)
            # print("\t\tself.indices", self.indices)
            # print("\t\tself.coefficients", type(self.coefficients))
            # print("\t\tself.indices", type(self.indices))

            self.indices.append(index)
            self.coefficients = np.append(self.coefficients, 0)

            # print("\t\tself.coefficients updated", self.coefficients)
            # print("\t\tself.indices updated", self.indices)

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

    def complete_iteration(self, energy, total_norm=None, sel_gradients=None):

        energy_change = energy - self.energy
        self.energy = energy

        # Save iteration data
        self.data.process_iteration(
            self.indices,
            self.energy,
            total_norm,
            sel_gradients,
            self.coefficients,
            # 0,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
            self.energy_opt_iters,
            self.shots_iters
        )

        # Update current state
        print("\n # Complete Iteration")
        print("\tCurrent energy:", self.energy, "change of", energy_change)
        print(f"\tCurrent ansatz: {list(self.indices)}")

        return    

    def optimize(self, gradient):
        """gradient: gradient of the last-added operator"""

        # Full Optimization
        print("\n # Standard VQE Optimization")

        self.cost_history_dict = {
            "prev_vector": None,
            "iters":0,
            "cost_history":[],
            'shots':[]
        }
        
        initial_coefficients = deepcopy(self.coefficients)
        indices = self.indices.copy()
        g0 = self.gradients
        e0 = self.energy
        maxiters = self.max_opt_iter

        print("\n\tEnergy Optimization Parameter:")
        print("\t\tInitial Coefficients:", initial_coefficients)
        print("\t\tIndices:", indices)
        print("\t\tg0:", g0)
        print("\t\te0:", e0)

        # parameters = ParameterVector("theta", len(indices))
        # qc = self.pool.get_circuit(indices, initial_coefficients, parameters)
        # ansatz = self.ref_circuit.compose(qc)

        # print("\tAnsatz Circuit:\n", ansatz)

        # print(
        #     f"\n\t// Optimization Property"
        #     f"\n\t\tInitial energy: {self.energy}"
        #     f"\n\t\tExact energy: {self.exact_energy}"
        #     f"\n\t\tError: {(self.exact_energy-self.energy)*627.5094} kcal/mol"
        #     f"\n\t\tOptimizing energy with indices {list(indices)}..."
        #     f"\n\t\tStarting point: {list(initial_coefficients)}"
        #     # f"\n\tNumber of Parameters: {ansatz.num_parameters}"
        #     f"\n\tIterations:"
        # )

        # Scipy Minimize
        res = minimize(
            self.evaluate_energy,
            initial_coefficients,
            args=(indices),
            method=self.optimizer_method,
        )

        # Qiskit Minimize
        # adam_optimizer = SPSA()
        # print("indices:", indices)
        # res = adam_optimizer.minimize(
        #     fun=self.evaluate_energy,
        #     x0=initial_coefficients,
        #     # args=(indices),
        # )

        print("\nScipy Optimize Result:",res)
        # energy = self.cost_history_dict['cost_history'][-1]

        
        self.coefficients = res.x
        print("\tself.coefficients updated:", self.coefficients)
        opt_energy = res.fun

        print("\nOptimized Circuit with Coefficients")
        print("Optimization Iteration at ADAPT-VQE Iter:", self.data.iteration_counter,":\n", self.cost_history_dict['cost_history'])
        # qc = self.pool.get_circuit_unparameterized(self.indices, self.coefficients)
        # self.qc_optimized = self.reference_circuit.compose(qc)
        # print(self.qc_optimized)

        print("\nCoefficients and Indices")
        print(f"\n\tError Percentage: {(self.exact_energy - opt_energy)/self.exact_energy*100}")
        print("\tself.coefficients initial:", self.coefficients)
        print("\tself.indices:", self.indices)

        self.energy_opt_iters = self.cost_history_dict['cost_history']
        self.shots_iters = self.cost_history_dict['shots']

        return opt_energy
    
    
    def evaluate_energy(self, coefficients=None, indices=None):

        ## Qiskit Estimator
        self.qiskit_hamiltonian = to_qiskit_operator(self.qubit_hamiltonian)

        if indices is None or coefficients is None: 
            # indices = []
            # indices = self.indices
            # print("--Indices:", indices)
            ansatz = self.ref_circuit
            pub = (ansatz, [self.qiskit_hamiltonian])

        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz = self.ref_circuit.compose(ansatz)
            pub = (ansatz, [self.qiskit_hamiltonian], [coefficients])


        result = self.estimator.run(pubs=[pub]).result()
              
        energy_qiskit_estimator = result[0].data.evs[0]
        print(f"\n\t> Opt Iteration-{self.cost_history_dict['iters']}")
        print("\n\t>> Qiskit Estimator Energy Evaluation")
        print(f"\t\tenergy_qiskit_estimator: {energy_qiskit_estimator} mHa,   c.a.e = {np.abs(energy_qiskit_estimator-self.exact_energy)*627.5094} kcal/mol")


        print(f"\n\t>> Qiskit Sampler Energy Evaluation ")
        if indices is None or coefficients is None:
            # indices = []
            # indices = self.indices
            ansatz = self.ref_circuit
        else:
            parameters = ParameterVector("theta", len(indices))
            ansatz = self.pool.get_parameterized_circuit(indices, coefficients, parameters)
            ansatz = self.ref_circuit.compose(ansatz)
    
        if self.shots_assignment == 'uniform':
            shots = self.uniform_shots_distribution(self.shots_budget, len(self.commuted_hamiltonian))
        else:
            shots = self.variance_shots_distribution(self.shots_budget, self.k, coefficients, ansatz)

        ansatz_cliques = []
        energy_qiskit_sampler = 0.0
        
        for i, cliques in enumerate(self.commuted_hamiltonian):

            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)

            ansatz_clique.measure_all()

            ansatz_cliques.append(ansatz_clique)

            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots = shots[i])

            counts = job.result()[0].data.meas.get_counts()

            probs = self.get_probability_distribution(counts, shots[i], self.n)

            for pauli_string in cliques:
                eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                
                res = np.dot(eigen_value, probs) * pauli_string.coeffs
                
                energy_qiskit_sampler += res[0].real

        print(f"\t\tenergy_qiskit_sampler: {energy_qiskit_sampler} mHa,   c.a.e = {np.abs(energy_qiskit_sampler-self.exact_energy)*627.5094} kcal/mol")

        self.cost_history_dict['iters'] += 1
        self.cost_history_dict['previous_vector'] = coefficients

        self.cost_history_dict['cost_history'].append(energy_qiskit_sampler)
        self.cost_history_dict['shots'].append(shots)

        error_chemac = np.abs(energy_qiskit_estimator - self.exact_energy) * 627.5094
        if error_chemac > 1:
            self.shots_chemac += np.sum(shots)
        print(f"\t\tAccumulated shots up to c.a.e: {self.shots_chemac} -> recent: {np.sum(shots)} {shots}")
  
        print("Return value:")
        print("Estimator", energy_qiskit_estimator)
        print("Sampler", energy_qiskit_sampler)

        return energy_qiskit_estimator
        # return energy_qiskit_sampler
    

    def uniform_shots_distribution(self, N, l):
        shots = [ N // l ] * l
        for i in range(N % l): shots[i] += 1
        return shots
    
    def variance_shots_distribution(self, N, k, coefficients, ansatz):

        ansatz_cliques = []

        std_cliques = []
        for i, cliques in enumerate(self.commuted_hamiltonian):
            # print(cliques)
            ansatz_clique = ansatz.copy()
            for j, pauli in enumerate(cliques[0].paulis[0]):
                if (pauli == self.PauliY):
                    ansatz_clique.sdg(j)
                    ansatz_clique.h(j)
                elif (pauli == self.PauliX):
                    ansatz_clique.h(j)
            ansatz_clique.measure_all()
            ansatz_cliques.append(ansatz_clique)

            job = self.sampler.run(pubs=[(ansatz_clique, coefficients)], shots = k)

            bitstrings = job.result()[0].data.meas.get_bitstrings()
            # print("Bitstirings:",bitstrings)

            results_array = self.convert_bitstrings_to_arrays(bitstrings, self.n)

            results_one_clique = []
            for m, count_res in enumerate(results_array):
                # print(f"\nResults of shot-{m+1}")
                # print(count_res)
                exp_pauli_clique = []
                for pauli_string in cliques:
                    eigen_value = self.get_eigenvalues(pauli_string.to_list()[0][0])
                    res = np.dot(eigen_value, count_res) * pauli_string.coeffs
                    exp_pauli_clique.append(res[0].real)
                # print(exp_pauli_clique)
                # print(np.sum(exp_pauli_clique))
                results_one_clique.append(np.sum(exp_pauli_clique))
            
            # print(f"\nResults of Clique-{i}", results_one_clique)
            # print(f"\nSTD of Clique-{i}", np.std(results_one_clique))
            std_cliques.append(np.std(results_one_clique))

        # print("\t\tSTD:", std_cliques)

        if sum(std_cliques) == 0:
            ratio_for_theta = [1/3 for _ in std_cliques]
        else:
            ratio_for_theta = [ v/sum(std_cliques) for v in std_cliques]
        
        # print("\t\tRatio for Theta", ratio_for_theta)


        # Shots Assignment Equations
        if self.shots_assignment == 'vmsa':
            new_shots_budget = (self.shots_budget - k*len(std_cliques))
        elif self.shots_assignment == 'vpsr':
            new_shots_budget = (self.shots_budget - k*len(std_cliques))*sum(ratio_for_theta)**2/3/sum([v**2 for v in ratio_for_theta])
        
        # print("\t\tNew Shots budget:",new_shots_budget)
        new_shots = [max(1, round(new_shots_budget * ratio_for_theta[i])) for i in range(len(std_cliques))]

        # print(new_shots)

        return new_shots
    
    def convert_bitstrings_to_arrays(self, bitstrings, N):
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

        

    def get_probability_distribution(self, counts, NUM_SHOTS, N):
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
    

    def get_eigenvalues(self, pauli_strings):
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
    