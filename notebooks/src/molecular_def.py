from openfermion import MolecularData
from openfermionpyscf import run_pyscf

# H2
geometry = [['H',[0,0,0]],['H',[0,0,0.74]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0
h2 = MolecularData(geometry,basis,multiplicity,charge,description='H2')
h2m = run_pyscf(h2,run_fci = True,run_ccsd = True)

# HeH+
r = 1 # interatomic distance in angstrom
geometry = [['He',[0,0,0]],['H',[0,0,r]]]
basis = 'sto-3g'
multiplicity = 1
charge = +1
helonium = MolecularData(geometry,basis,multiplicity,charge,description='HeH+')
helonium = run_pyscf(helonium,run_fci = True,run_ccsd = True)

# LiH
bondLength = 1.45 # interatomic distance in angstrom
geometry = [['Li',[0,0,0]],['H',[0,0,bondLength]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0
liH = MolecularData(geometry,basis,multiplicity,charge,description='LiH')
liH = run_pyscf(liH,run_fci = True,run_ccsd = True)

# Alternative to using run_pyscf: load from OpenFermion (data for this 
#particular molecule at this particular interatomic distance is available in 
#a file that comes with OF)
liHOF = MolecularData(geometry,basis,multiplicity,charge,description = '1.45')
liHOF.load()

# H4
r = 1.5
geometry = [('H', (0,0,0)), ('H', (0,0,r)), ('H', (0,0,2*r)), 
            ('H', (0,0,3*r))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
h4 = MolecularData(geometry,basis,multiplicity,charge,description='H4')
h4 = run_pyscf(h4,run_fci = True,run_ccsd = True)

# H6
r = 1.5
geometry = [('H', (0,0,0)), ('H', (0,0,r)), ('H', (0,0,2*r)), 
            ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
h6 = MolecularData(geometry,basis,multiplicity,charge,description='H6')
h6 = run_pyscf(h6,run_fci = True,run_ccsd = True)