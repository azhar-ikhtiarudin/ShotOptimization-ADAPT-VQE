from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.hamiltonian import h_lih
from algorithms.adapt_vqe_v8 import AdaptVQE


if __name__ == '__main__':    
    r = 1.45
    molecule = create_lih(r)
    print(molecule.fci_energy)
    pool = QE(molecule=None,
            frozen_orbitals=[],
            n=4)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=None,
                        max_adapt_iter=50,
                        max_opt_iter=100,
                        grad_threshold=1e-3,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=100,
                        shots_budget=1024,
			N_experiments=1000,
                        backend_type='noisy',
                        custom_hamiltonian=h_lih,
                        noise_level=0.001
                        )

    adapt_vqe.run()
