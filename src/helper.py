import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy

from openfermion.utils import count_qubits, hermitian_conjugated
from openfermionpyscf import run_pyscf
from openfermion.utils import count_qubits, hermitian_conjugated
from openfermion.linalg import jw_hartree_fock_state
from openfermion.circuits import simulate_trotter
from openfermion.transforms import (
    jordan_wigner, get_fermion_operator, normal_ordered
)
from openfermion import (
    get_sparse_operator, get_ground_state, FermionOperator,
    jw_get_ground_state_at_particle_number, MolecularData,
    expectation, uccsd_convert_amplitude_format,
    get_interaction_operator, QubitOperator, eigenspectrum,
    InteractionOperator, FermionOperator
)

from qiskit_algorithms import NumPyMinimumEigensolver


from qiskit import (QuantumCircuit, ClassicalRegister, QuantumRegister)
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter

from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
# from qiskit.providers.ibmq import least_busy

from qiskit_aer import QasmSimulator
from qiskit.quantum_info import purity
from qiskit.quantum_info import SparsePauliOp
# from qiskit_aer.noise.errors.standard_error import pauli_error
# from qiskit_aer.noise import pauli_error,gate_error_values
# from qiskit_aer.noise import pauli_error
# from qiskit_aer.noise import (basic_device_gate_errors, gate_error_values, NoiseModel,thermal_relaxation_error, pauli_error, ReadoutError)

from qiskit_ibm_runtime.fake_provider import FakeBelem
from qiskit_aer import QasmSimulator

# from qiskit.aqua import QuantumInstance
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

from qiskit.primitives import Estimator


# Define chemical accuracy
chemicalAccuracy = 1.5936*10**-3

# Define necessary Pauli operators (two-dimensional) as matrices
pauliX = np.array([[0,1],
                 [1,0]],
                dtype = complex)
pauliZ = np.array([[1,0],
                 [0,-1]],
                dtype = complex)
pauliY = np.array([[0,-1j],
                 [1j,0]],
                dtype = complex)

def stringToMatrix(pauliString):
  '''
  Converts a Pauli string to its matrix form.

  Arguments:
    pauliString (str): the Pauli string (e.g. "IXYIZ")

  Returns:
    matrix (np.ndarray): the corresponding matrix, in the computational basis

  '''

  matrix = np.array([1])

  # Iteratively construct the matrix, going through each single qubit Pauli term
  for pauli in pauliString:
      if pauli == "I":
        matrix = np.kron(matrix,np.identity(2))
      elif pauli == "X":
        matrix = np.kron(matrix,pauliX)
      elif pauli == "Y":
        matrix = np.kron(matrix,pauliY)
      elif pauli == "Z":
        matrix = np.kron(matrix,pauliZ)

  return matrix
  
def fromVectortoKet(stateVector):
  '''
  Transforms a vector representing a basis state to the corresponding ket.

  Arguments:
    stateVector (np.ndarray): computational basis vector in the 2^n dimensional 
      Hilbert space

  Returns:
    ket (list): a list of length n representing the corresponding ket 
  '''

  dim = len(stateVector)
  ket = []

  while dim>1:
    if any (stateVector[i] for i in range(int(dim/2))):
      # Ket is of the form |0>|...>. 

      #Fix |0> as the msq.
      ket.append(0)

      # Get the vector representing the state of the remaining qubits.
      stateVector = stateVector[:int(dim/2)]

    else:
      # Ket is of the form |1>|...>. 
      
      #Fix |0> as the msq.
      ket.append(1)

      # Get the vector representing the state of the remaining qubits.
      stateVector = stateVector[int(dim//2):]

    dim = dim/2

  return ket

def fromKettoVector(ket):
  '''
  Transforms a ket representing a basis state to the corresponding state vector.

  Arguments:
    ket (list): a list of length n representing the ket 

  Returns:
    stateVector (np.ndarray): the corresponding basis vector in the 
      2^n dimensional Hilbert space
  '''
  stateVector = [1]
  
  # Iterate through the ket, calculating the tensor product of the qubit states
  for i in ket:
    qubitVector = [not i,i]
    stateVector = np.kron(stateVector,qubitVector)

  return stateVector

def slaterDeterminantToKet(index, dimension):
  '''
  Transforms a Slater Determinant (computational basis state) into
    the corresponding ket.

  Arguments:
    index (int): the index of the non-zero element of the computational
      basis state.
    dimension (int): the dimension of the Hilbert space
  
  Returns:
    ket (list): the corresponding ket as a list of length dimension
    
  '''
  
  vector = [0 for _ in range (index)]+[1]+[1 for _ in range (dimension-index-1)]
  ket = fromVectortoKet(vector)

  return ket
  
def getHartreeFockKet(electronNumber,qubitNumber):
  '''
  Get the Hartree Fock determinant, as a list in big endian representing the ket
  |1>|1>...|0>|0>.

  Arguments:
    electronNumber (int): the number of electrons of the molecule.
    qubitNumber (int): the number of qubits necessary to represent the molecule
      (equal to the number of spin orbitals we're considering active).

  Returns:
    referenceDeterminant (list): a list of lenght qubitNumber, representing the 
      ket of the adequate computational basis state in big-endian ordering.
  '''

  # Consider occupied the lower energy orbitals, until enough one particle 
  #states are filled
  referenceDeterminant = [1 for _ in range(electronNumber)]

  # Consider the remaining orbitals empty
  referenceDeterminant += [0 for _ in range(qubitNumber-electronNumber)]

  return referenceDeterminant

def calculateOverlap(stateCoordinates1,stateCoordinates2):
    '''
    Calculates the overlap between two states, given their coordinates.

    Arguments:
      stateCoordinates1 (np.ndarray): the coordinates of one of the states in 
        some orthonormal basis,
      stateCoordinates2 (np.ndarray): the coordinates of the other state, in 
        the same basis

    Returns: 
      overlap (float): the overlap between two states (absolute value of the 
        inner product).
    '''

    bra = np.conj(stateCoordinates1)
    ket = stateCoordinates2
    overlap = np.abs(np.dot(bra,ket))
    
    return overlap

def findSubStrings(mainString,hamiltonian,checked = []):
    '''
    Finds and groups all the strings in a Hamiltonian that only differ from 
    mainString by identity operators.

    Arguments:
      mainString (str): a Pauli string (e.g. "XZ)
      hamiltonian (dict): a Hamiltonian (with Pauli strings as keys and their 
        coefficients as values)
      checked (list): a list of the strings in the Hamiltonian that have already
        been inserted in another group

    Returns: 
      groupedOperators (dict): a dictionary whose keys are boolean strings 
        representing substrings of the mainString (e.g. if mainString = "XZ", 
        "IZ" would be represented as "01"). It includes all the strings in the 
        hamiltonian that can be written in this form (because they only differ 
        from mainString by identities), except for those that were in checked
        (because they are already part of another group of strings).
      checked (list):  the same list passed as an argument, with extra values
        (the strings that were grouped in this function call).
    '''
    
    groupedOperators = {}
    
    # Go through the keys in the dictionary representing the Hamiltonian that 
    #haven't been grouped yet, and find those that only differ from mainString 
    #by identities
    for pauliString in hamiltonian:
        
        if pauliString not in checked:
            # The string hasn't been grouped yet
            
            if(all((op1 == op2 or op2 == "I") \
                   for op1,op2 in zip(mainString,pauliString))):
                # The string only differs from mainString by identities
                
                # Represent the string as a substring of the main one
                booleanString = "".join([str(int(op1 == op2)) for op1,op2 in \
                                       zip(mainString,pauliString)])
                    
                # Add the boolean string representing this string as a key to 
                #the dictionary of grouped operators, and associate its 
                #coefficient as its value
                groupedOperators[booleanString] = hamiltonian[pauliString]
                
                # Mark the string as grouped, so that it's not added to any 
                #other group
                checked.append(pauliString)
                
    return (groupedOperators,checked)

def groupHamiltonian(hamiltonian):
    '''
    Organizes a Hamiltonian into groups where strings only differ from 
    identities, so that the expectation values of all the strings in each 
    group can be calculated from the same measurement array.

    Arguments: 
      hamiltonian (dict): a dictionary representing a Hamiltonian, with Pauli 
        strings as keys and their coefficients as values.

    Returns: 
      groupedHamiltonian (dict): a dictionary of subhamiltonians, each of 
        which includes Pauli strings that only differ from each other by 
        identities. 
        The keys of groupedHamiltonian are the main strings of each group: the 
        ones with least identity terms. The value associated to a main string is 
        a dictionary, whose keys are boolean strings representing substrings of 
        the respective main string (with 1 where the Pauli is the same, and 0
        where it's identity instead). The values are their coefficients.
    '''
    groupedHamiltonian = {}
    checked = []
    
    # Go through the hamiltonian, starting by the terms that have less
    #identity operators
    for mainString in \
        sorted(hamiltonian,key = lambda pauliString: pauliString.count("I")):
            
        # Call findSubStrings to find all the strings in the dictionary that 
        #only differ from mainString by identities, and organize them as a 
        #dictionary (groupedOperators)
        groupedOperators,checked = findSubStrings(mainString,hamiltonian,checked)
        
        # Use the dictionary as a value for the mainString key in the 
        #groupedHamiltonian dictionary
        groupedHamiltonian[mainString] = groupedOperators
        
        # If all the strings have been grouped, exit the for cycle
        if(len(checked) == len(hamiltonian.keys())):
           break
       
    return groupedHamiltonian

def convertHamiltonian(openfermionHamiltonian):
  '''
  Formats a qubit Hamiltonian obtained from openfermion, so that it's a suitable
  argument for functions such as measureExpectationEstimation.

  Arguments:
    openfermionHamiltonian (openfermion.qubitOperator): the Hamiltonian.

  Returns:
    formattedHamiltonian (dict): the Hamiltonian as a dictionary with Pauli
      strings (eg 'YXZI') as keys and their coefficients as values.
  '''

  formattedHamiltonian = {}
  qubitNumber = count_qubits(openfermionHamiltonian)

  # Iterate through the terms in the Hamiltonian
  for term in openfermionHamiltonian.get_operators():

    operators = []
    coefficient = list(term.terms.values())[0]
    pauliString = list(term.terms.keys())[0]
    previousQubit = -1

    for (qubit,operator) in pauliString:

      # If there are qubits in which no operations are performed, add identities 
      #as necessary, to make sure that the length of the string will match the 
      #number of qubits
      identities = (qubit-previousQubit-1)
      if identities>0: 
        operators.append('I'*identities)

      operators.append(operator)
      previousQubit = qubit
    
    # Add final identity operators if the string still doesn't have the 
    #correct length (because no operations are performed in the last qubits)
    operators.append('I'*(qubitNumber-previousQubit-1))

    formattedHamiltonian["".join(operators)] = coefficient

    # print('===List Hamiltonian===')
    # for key, value in formattedHamiltonian.items():
    #    print(key, value)

  return formattedHamiltonian

def hamiltonianToMatrix(hamiltonian):
    '''
    Convert a Hamiltonian (from OpenFermion) to matrix form.
    
    Arguments:
      hamiltonian (openfermion.InteractionOperator): the Hamiltonian to be
        transformed.

    Returns:
      matrix (np.ndarray): the Hamiltonian, as a matrix in the computational 
        basis
    
    ''' 
    
    qubitNumber = hamiltonian.n_qubits    
    hamiltonian = jordan_wigner(hamiltonian)

    formattedHamiltonian = convertHamiltonian(hamiltonian)
    print('Formatted Hamiltonian:', formattedHamiltonian)
    groupedHamiltonian = groupHamiltonian(formattedHamiltonian)

    matrix = np.zeros((2**qubitNumber,2**qubitNumber),dtype = complex)

    # Iterate through the strings in the Hamiltonian, adding the respective 
    #contribution to the matrix
    for string in groupedHamiltonian:

      for substring in groupedHamiltonian[string]:
        pauli = ("".join("I"*(not int(b)) + a*int(b) \
                         for (a,b) in zip(string,substring)))
        
        matrix += stringToMatrix(pauli) * groupedHamiltonian[string][substring]

    return matrix

def stateEnergy(stateCoordinates,hamiltonian):
    ''' 
    Calculates the exact energy in a specific state.

    Arguments:
      stateCoordinates (np.ndarray): the state in which to obtain the 
        expectation value.
      hamiltonian (dict): the Hamiltonian of the system.
    
    Returns:
      exactEnergy (float): the energy expecation value in the state.
    ''' 

    exactEnergy = 0
    
    # Obtain the theoretical expectation value for each Pauli string in the
    #Hamiltonian by matrix multiplication, and perform the necessary weighed
    #sum to obtain the energy expectation value.
    for pauliString in hamiltonian:
        
        ket = np.array(stateCoordinates,dtype = complex)
        bra = np.conj(ket)
        
        ket = np.matmul(stringToMatrix(pauliString),ket)
        expectationValue = np.real(np.dot(bra,ket))
        
        exactEnergy+=\
            hamiltonian[pauliString]*expectationValue
            
    return exactEnergy

def exactStateEnergySparse(stateVector,sparseHamiltonian):
    ''' 
    Calculates the exact energy in a specific state, using sparse matrices.

    Arguments:
      stateVector (Union[np.ndarray, scipy.sparse.csc_matrix): the state in 
        which to obtain the expectation value.
      sparseHamiltonian (scipy.sparse.csc_matrix): the Hamiltonian of the system.
    
    Returns:
      energy (float): the energy expecation value in the state.
    ''' 

    if not isinstance(stateVector,scipy.sparse.csc_matrix):
      ket = scipy.sparse.csc_matrix(stateVector,dtype=complex).transpose()
    else:
      ket = stateVector
      
    bra = ket.transpose().conj()

    energy = (bra * sparseHamiltonian * ket)[0,0].real
    
    return energy

def pauliToCircuit(operator,parameter,circuit,qubitNumber): 
  '''
  Creates the circuit for applying e^ (j * operator * parameter), for 'operator'
  a single Pauli string.
  Uses little endian endian, as Qiskit requires.

  Arguments:
    operator (union[openfermion.QubitOperator, openfermion.FermionOperator,
      openfermion.InteractionOperator]): the operator to be simulated
    parameter (qiskit.circuit.parameter): the variational parameter
    circuit (qiskit.circuit.QuantumCircuit): the circuit to add the gates to
    qubitNumber (int): the number of qubits in the circuit
  '''

  # If operator is an InteractionOperator, shape it into a FermionOperator
  if isinstance(operator,InteractionOperator):
    operator = get_fermion_operator(operator)

  # If operator is a FermionOperator, use the Jordan Wigner transformation
  #to map it into a QubitOperator
  if isinstance(operator,FermionOperator):
    operator = jordan_wigner(operator)

  # Isolate the Pauli string (term should only have one)
  pauliString = list(operator.terms.keys())[0]

  # Keep track of the qubit indices involved in this particular Pauli string.
  # It's necessary so as to know which are included in the sequence of CNOTs 
  #that compute the parity
  involvedQubits = []

  # Perform necessary basis rotations
  for pauli in pauliString:

    # Get the index of the qubit this Pauli operator acts on
    qubitIndex = pauli[0]
    involvedQubits.append(qubitIndex)

    # Get the Pauli operator identifier (X,Y or Z)
    pauliOp = pauli[1]

    if pauliOp == "X":
      # Rotate to X basis
      # In big endian the argument would be qubitIndex
      circuit.h(qubitNumber - 1 - qubitIndex)

    if pauliOp == "Y":
      # Rotate to Y Basis
      circuit.rx(np.pi/2,qubitNumber - 1 - qubitIndex)

  # Compute parity and store the result on the last involved qubit
  for i in range(len(involvedQubits)-1):

    control = involvedQubits[i]
    target = involvedQubits[i+1]

    circuit.cx(qubitNumber - 1 - control,qubitNumber - 1 - target)
  
  # Apply e^(i*Z*parameter) = Rz(-parameter*2) to the last involved qubit
  lastQubit = max(involvedQubits)
  circuit.rz(-2 * parameter,qubitNumber - 1 - lastQubit)

  # Uncompute parity
  for i in range(len(involvedQubits)-2,-1,-1):

    control = involvedQubits[i]
    target = involvedQubits[i+1]

    circuit.cx(qubitNumber - 1 - control,qubitNumber - 1 - target)

  # Undo basis rotations
  for pauli in pauliString:

    # Get the index of the qubit this Pauli operator acts on
    qubitIndex = pauli[0]

    # Get the Pauli operator identifier (X,Y or Z)
    pauliOp = pauli[1]

    if pauliOp == "X":
      # Rotate to Z basis from X basis
      circuit.h(qubitNumber - 1 - qubitIndex)

    if pauliOp == "Y":
      # Rotate to Z basis from Y Basis
      circuit.rx(-np.pi/2,qubitNumber - 1 - qubitIndex)

def getObservable(operator):
  observable = 0

  for pauliString in operator:

    transformedPauli = 1
    for pauli in pauliString:
      if pauli == "I":
         print("I")
        # transformedPauli = transformedPauli ^ IGate
      elif pauli == "X":
         print("X")
        # transformedPauli = transformedPauli ^ XGate
      elif pauli == "Y":
         print("Y")
        # transformedPauli = transformedPauli ^ YGate
      elif pauli == "Z":
         print("Z")
        # transformedPauli = transformedPauli ^ ZGate


    coefficient = operator[pauliString]
    print("Coefficient: ", coefficient)
    observable += transformedPauli * coefficient

  return observable