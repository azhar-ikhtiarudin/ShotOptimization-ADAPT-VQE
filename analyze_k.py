from src.pools import QE
from src.molecules import create_h2, create_h3

from algorithms.adapt_vqe_v2 import AdaptVQE

r = 0.742
molecule = create_h2(r)
pool = QE(molecule)

adapt_vqe = AdaptVQE(pool=pool,
                    molecule=molecule,
                    max_adapt_iter=1,
                    max_opt_iter=100,
                    grad_threshold=10**-5,
                    vrb=True,
                    optimizer_method='nelder-mead',
                    shots_assignment='vpsr',
                    k=10,
                    shots_budget=1000,
                    N_experiments=100
                    # seed=3
                    )


# adapt_vqe.analyze_k()
adapt_vqe.analyze_k()