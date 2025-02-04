from src.pools import QE
from src.molecules import create_h2, create_h3, create_h4

from algorithms.adapt_vqe_v7 import AdaptVQE


if __name__ == '__main__':    
    
    r = 1.75
    molecule = create_h2(r)
    pool = QE(molecule)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=molecule,
                        max_adapt_iter=10,
                        max_opt_iter=100,
                        grad_threshold=1e-3,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=100,
                        shots_budget=1024,
                        N_experiments=1000,
                        backend_type='noiseless',
                        custom_hamiltonian=None
                        )

    adapt_vqe.run()
