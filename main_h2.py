from src.pools import QE
from src.molecules import create_h2, create_h3, create_h4

from algorithms.adapt_vqe_v6_h2_fixed import AdaptVQE


if __name__ == '__main__':    
    r = 1.75
    molecule = create_h2(r)
    pool = QE(molecule=None,
            frozen_orbitals=[],
            n=4)
    
    print("4 qubits pool", pool.__dict__)

    pool = QE(molecule=None,
            frozen_orbitals=[],
            n=2)
    
    print("2 qubits pool", pool.__dict__)
    print(molecule.fci_energy)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=molecule,
                        max_adapt_iter=10,
                        max_opt_iter=100,
                        grad_threshold=1e-15,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=100,
                        shots_budget=5000,
                        N_experiments=50,
                        backend_type='aer-default'
                        )

    adapt_vqe.run()
