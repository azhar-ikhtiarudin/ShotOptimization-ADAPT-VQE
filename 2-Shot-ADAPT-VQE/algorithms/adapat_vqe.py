import copy
import AdaptData
import numpy as np

class AdaptVQE():
    """
        Main Class for ADAPT-VQE Algorithm
    """

    def __init__(self, pool, molecule, max_adapt_iter, max_opt_iter, grad_threshold=10**-8, vrb=False):
        self.pool = pool
        self.molecule = copy(molecule)
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.vrb = vrb
        self.grad_threshold = grad_threshold

    def run(self):
        if self.vrb: print("Run, Initializing . . .")
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            finished = self.run_iteration()


    def initialize(self):
        self.initial_energy = self.evaluate_energy()        
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





