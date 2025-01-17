from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih

from algorithms.adapt_vqe_v3 import AdaptVQE


if __name__ == '__main__':    
    r = 1.595
    molecule = create_lih(r)
    pool = QE(molecule)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=molecule,
                        max_adapt_iter=50,
                        max_opt_iter=100,
                        grad_threshold=0.01,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=1000,
                        shots_budget=10000000,
			N_experiments=20
                        )

    adapt_vqe.run()
