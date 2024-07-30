from openfermion import (
    get_sparse_operator, get_ground_state, FermionOperator,
    jw_get_ground_state_at_particle_number, MolecularData,
    expectation, uccsd_convert_amplitude_format,
    get_interaction_operator, QubitOperator, eigenspectrum,
    InteractionOperator, FermionOperator
)
from openfermion.utils import count_qubits, hermitian_conjugated
from openfermion.transforms import (
    jordan_wigner, get_fermion_operator, normal_ordered
)

import numpy as np

def fermionicPool(orbitalNumber):
    singlet_gsd = []

    for p in range(0,orbitalNumber):
        pa = 2*p
        pb = 2*p+1

        for q in range(p,orbitalNumber):
            qa = 2*q
            qb = 2*q+1

            termA =  FermionOperator(((pa,1),(qa,0)))
            termA += FermionOperator(((pb,1),(qb,0)))

            termA -= hermitian_conjugated(termA)
            termA = normal_ordered(termA)

            #Normalize
            coeffA = 0
            for t in termA.terms:
                coeff_t = termA.terms[t]
                coeffA += coeff_t * coeff_t

            if termA.many_body_order() > 0:
                termA = termA/np.sqrt(coeffA)
                singlet_gsd.append(termA)


    pq = -1
    for p in range(0,orbitalNumber):
        pa = 2*p
        pb = 2*p+1

        for q in range(p,orbitalNumber):
            qa = 2*q
            qb = 2*q+1

            pq += 1

            rs = -1
            for r in range(0,orbitalNumber):
                ra = 2*r
                rb = 2*r+1

                for s in range(r,orbitalNumber):
                    sa = 2*s
                    sb = 2*s+1

                    rs += 1

                    if(pq > rs):
                        continue

                    termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                    termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                    termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                    termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                    termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                    termA -= hermitian_conjugated(termA)
                    termB -= hermitian_conjugated(termB)

                    termA = normal_ordered(termA)
                    termB = normal_ordered(termB)

                    #Normalize
                    coeffA = 0
                    coeffB = 0
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
                    for t in termB.terms:
                        coeff_t = termB.terms[t]
                        coeffB += coeff_t * coeff_t


                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        singlet_gsd.append(termA)

                    if termB.many_body_order() > 0:
                        termB = termB/np.sqrt(coeffB)
                        singlet_gsd.append(termB)

    print("Pool size:",len(singlet_gsd))
    return singlet_gsd

def qubitPool(singlet_gsd):
    pool = singlet_gsd
    qubitPool = []

    for fermionOp in pool:
        # print("fermionOp:", fermionOp, "\n")
        qubitOp = jordan_wigner(fermionOp)
        # print('qubitOp:', qubitOp, "\n")
        
        for pauli in qubitOp.terms:
            qubitOp = QubitOperator(pauli,1j)
            # print('qubitOp 2:', qubitOp, "\n")

            if qubitOp not in qubitPool:
                qubitPool.append(qubitOp)

    print("Pool Size:",len(qubitPool))
    
    return qubitPool