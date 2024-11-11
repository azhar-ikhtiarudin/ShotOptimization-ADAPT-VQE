from openfermion import MolecularData
from openfermionpyscf import run_pyscf

def create_h2(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h2 (PyscfMolecularData): the linear H2 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h2 = MolecularData(geometry, basis, multiplicity, charge, description='H2')
    h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)

    return h2


def create_h3(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h3 (PyscfMolecularData): the linear H3 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]], ['H', [0, 0, 2 * r]]]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    h3 = MolecularData(geometry, basis, multiplicity, charge, description='H3')
    h3 = run_pyscf(h3, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?

    return h3


def create_h4(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h4 (PyscfMolecularData): the linear H4 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
                ('H', (0, 0, 3 * r))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h4 = MolecularData(geometry, basis, multiplicity, charge, description='H4')
    h4 = run_pyscf(h4, run_fci=True, run_ccsd=True)

    return h4