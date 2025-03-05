from src.pools import QE, NoZPauliPool
from src.molecules import create_h2, create_h3, create_h4, create_lih

from algorithms.adapt_vqe_v10_meas_recycle import AdaptVQE


if __name__ == '__main__':    
    
    r = 0.742
    molecule = create_lih(r)
    pool = QE(molecule)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=molecule,
                        max_adapt_iter=100,
                        max_opt_iter=100,
                        grad_threshold=1e-6,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=100,
                        shots_budget=1024,
                        N_experiments=10,
                        backend_type='noiseless',
                        custom_hamiltonian=None,
                        noise_level=0.00001
                        )

    adapt_vqe.run()