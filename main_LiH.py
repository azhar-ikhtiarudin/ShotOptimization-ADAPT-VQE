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
<<<<<<< HEAD
                        k=10000,
                        shots_budget=2000000000,
=======
                        k=10,
                        shots_budget=1000,
>>>>>>> 3f487fb0b9aed70b849fa707d33c7f8245686134
                        N_experiments=20
                        )

    adapt_vqe.run()
