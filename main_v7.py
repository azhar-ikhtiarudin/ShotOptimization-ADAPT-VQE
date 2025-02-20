from src.pools import QE
from src.molecules import create_h2, create_h3, create_lih
from src.hamiltonian import h_lih
from algorithms.adapt_vqe_v7 import AdaptVQE
from src.utilities import to_qiskit_operator


if __name__ == '__main__':    
    r = 1.45
    molecule = create_lih(r)
#     print(molecule.__dict__)
    # print(molecule.fci_energy)
    # pool = QE(molecule)
    pool = QE(molecule=None,
            frozen_orbitals=[],
            n=4)
    
    qiskit_hamiltonian = to_qiskit_operator(h_lih)
    print(qiskit_hamiltonian)

    # adapt_vqe = AdaptVQE(pool=pool,
    #                     molecule=None,
    #                     max_adapt_iter=50,
    #                     max_opt_iter=100,
    #                     grad_threshold=1e-3,
    #                     vrb=True,
    #                     optimizer_method='l-bfgs-b',
    #                     shots_assignment='uniform',
    #                     k=100,
    #                     shots_budget=10240,
	# 		            N_experiments=100,
    #                     backend_type='noisy',
    #                     custom_hamiltonian=h_lih
    #                     )

    # adapt_vqe.run()
