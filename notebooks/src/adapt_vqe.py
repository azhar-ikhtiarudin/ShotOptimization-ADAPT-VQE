from src.helper import *
# get_sparse_operator = 1
from qiskit_aer import QasmSimulator
from src.data_type import AdaptData


backend = QasmSimulator

class AdaptVQE:
  '''
  Class for running the VQE algorithm.

  Methods:
    prepare: to be called before the run, in order to prepare the sparse pool
      and possibly upload previous data
    printSettings: print the algorithm options
    computeState: calculate the state with the current ansatz and coefficients
    calculateOperatorGradient: calculate the absolute vale of the gradient of 
      a given operator
    selectOperator: select the operator to be added to the ansatz
    calculateEnergy: calculate the energy with a certain ansatz and coefficients
    callback: store intermediate data from the optimization
    optimizeCircuit: optimize a parametrized circuit with a given ansatz
    run: run the algorithm
  '''

  def __init__(self,
               pool,
               molecule,
               referenceDeterminant = None,
               verbose = False,
               maxIterations = 50,
               threshold = 0.1,
               shots = 1000):
    '''
    Initialize class instance

    Arguments:
      pool (list): operator pool
      molecule (openfermion.MolecularData): the molecule we're finding the 
        ground state of
      referenceDeterminant (list): the Slater determinant to be used as reference,
        in big endian ordering (|abcd> <-> [a,b,c,d]; [qubit 0, qubit 1,...]).
        If none, the Hartree Fock determinant will be used.
      verbose (bool): whether to print all the non-zero gradients, or to just
        leave out that information
      maxIterations (int): maximum allowed number of iterations until forcing
        a stop, even if the convergence threshold isn't met
      threshold (float): the convergence threshold. When the total gradient norm
        is lower than this, the algorithm will stop
      backend (Union[None,qiskit.providers.ibmq.IBMQBackend]): the backend to 
        be used. If none, a simulation will be run using sparse matrices.
      shots (int): number of circuit repetitions
    '''
    self.pool = pool.copy()
    self.molecule = molecule
    self.verbose = verbose

    self.referenceDeterminant = referenceDeterminant
    self.electronNumber = molecule.n_electrons

    self.hamiltonian = molecule.get_molecular_hamiltonian()
    self.qubitNumber = self.hamiltonian.n_qubits
    
    self.sparseHamiltonian = get_sparse_operator(self.hamiltonian,self.qubitNumber)
    self.qubitHamiltonian = jordan_wigner(self.hamiltonian)
    
    dictHamiltonian = convertHamiltonian(self.qubitHamiltonian)

    # print("Qubit Hamiltonian:", self.qubitHamiltonian)
    # print("Dict Hamiltonian:", dictHamiltonian)

    list_hamiltonian = list(dictHamiltonian.items())
    # print("Hamiltonian List", list_hamiltonian)

    self.hamiltonian = SparsePauliOp.from_list(list(dictHamiltonian.items()))
    # print("Final Hamiltonian: ", self.hamiltonian)

    numpy_solver = NumPyMinimumEigensolver()
    result = numpy_solver.compute_minimum_eigenvalue(operator=self.hamiltonian)
    ref_value = result.eigenvalue.real
    print(f"Reference value: {ref_value:.5f}")
    

    # self.observable = getObservable(dictHamiltonian)
    # print("Observable:", self.observable)
  
    # Format and group the Hamiltonian, so as to save measurements in the 
    #circuit simulation by using the same data for Pauli strings that only 
    #differ by identities

    # self.formattedHamiltonian = convertHamiltonian(self.qubitHamiltonian)
    # print('Formatted Hamiltonian:', self.formattedHamiltonian)
    # self.groupedHamiltonian = groupHamiltonian(self.formattedHamiltonian)
    # print('Grouped Hamiltonian:', self.groupedHamiltonian)

    self.maxIterations = maxIterations
    self.threshold = threshold

    self.backend = backend
    self.shots = shots

    self.sparsePool = []
    self.ansatz = []
    self.coefficients = []
    self.indices = []

  def prepare(self,
              sparsePool = None,
              previousData = None):
    '''
    Prepare to run the algorithm

    Arguments:
      sparsePool (list): a sparse version of the pool, to avoid reobtaining 
        sparse versions of the operators if they've already been obtained
      previousData (AdaptData): data from a previous run, that will be continued
      
    '''


    # If they weren't provided, obtain sparse versions of the operators to 
    #avoid constantly recalculating them
    if sparsePool is None:
      print("Sparse version of the pool was not provided. Obtaining it...")

      for operator in self.pool:
        self.sparsePool.append(get_sparse_operator(operator,self.qubitNumber))

    else:
      self.sparsePool = sparsePool
      

    print("Initializing data structures...")

    if self.referenceDeterminant is None:
      # Set the Hartree Fock state as reference
      self.referenceDeterminant = getHartreeFockKet(self.electronNumber,self.qubitNumber)

    print("\nAdapt VQE prepared with the following settings:")
    self.printSettings()
    self.referenceState = fromKettoVector(self.referenceDeterminant)

    self.sparseReferenceState = scipy.sparse.csc_matrix(
        self.referenceState,dtype=complex
        ).transpose()

    # Calculate energy of the reference state.
    initialEnergy = self.calculateEnergy([],[])

    # Initialize insance of AdaptData class that will store the data from the 
    #run
    self.data = AdaptData(initialEnergy,
                          self.pool,
                          self.sparsePool,
                          self.referenceDeterminant,
                          self.backend,
                          self.shots,
                          previousData)
        
    if self.data.current["state"] is None:
      # There's no previous data; initialize state at reference
      self.data.current["state"] = self.sparseReferenceState
    
  def printSettings(self):
      '''
      Prints the options that were chosen for the Adapt VQE run.
      '''

      print("> Convergence threshold (gradient norm): ", self.threshold)
      print("> Maximum number of iterations: ", self.maxIterations)
      print("> Backend: ",self.backend)
      # if self.backend is None:
      #   print("(using sparse matrices)")
      # elif backend.name() != "statevector_simulator":
      #   print("(Shot number: {})".format(self.shots))

  def computeState(self):
    '''
      Calculates the state with the current ansatz and coefficients.

      Returns:
        state (scipy.sparse.csc_matrix): the state
    '''

    # Since the coefficients are reoptimized every iteration, the state has to 
    #be built from the reference state each time.

    # Initialize the state vector with the reference state.
    state = self.sparseReferenceState

    # Apply the ansatz operators one by one to obtain the state as optimized
    #by the last iteration
    for (i,operatorIndex) in enumerate(self.data.current["indices"]):

      # Obtain the exponentiated pool operator, multiplied by the respective
      #variational parameter (as optimized by the last iteration)
      coefficient = self.data.current["coefficients"][i]
      sparseOperator = self.sparsePool[operatorIndex].multiply(coefficient)
      expOperator = scipy.sparse.linalg.expm(sparseOperator)

      # Act on the state with the operator 
      state = expOperator * state

    return state

  def calculateOperatorGradient(self,operatorIndex):
    '''
      Calculates the gradient of a given operator in the current state.
      Uses dexp(c*A)/dc = <psi|[H,A]|psi> = 2 * real(<psi|HA|psi>).
      This is the gradient calculated at c = 0, which will be the initial value 
      of the coefficient in the optimization.

      Arguments:
        operatorIndex (int): the index that labels this operator
      
      Returns:
        gradient (float): the norm of the gradient of this operator in the 
          current state
    '''

    sparseOperator = self.sparsePool[operatorIndex]
    currentState = self.data.current["state"]

    testState = sparseOperator * currentState
    bra = currentState.transpose().conj()

    gradient = 2 * (np.abs(bra * self.sparseHamiltonian * testState)[0,0].real)

    return gradient
  
  def selectOperator(self):
    ''' 
    Choose the next operator to be added to the ansatz, using as criterion
    that the one with the maximum gradient norm is the selected one.

    Returns:
      selectedIndex (int): the index that labels the selected operator
      selectedGradient (float): the norm of the gradient of that operator
      totalNorm (float): the total gradient norm
    '''

    selectedGradient = 0
    selectedIndex = None
    totalNorm = 0

    print("Calculating gradients and selecting the next operator...")
    
    if self.verbose:
      print("\nNon-Zero Gradients (calculated, tolerance E-5):")

    for operatorIndex in range(len(self.pool)):

      gradient = self.calculateOperatorGradient(operatorIndex)

      totalNorm += gradient**2

      if self.verbose:
        if gradient > 10**-5:
          print("Operator {}: {}".format(operatorIndex,gradient))

      if gradient > selectedGradient:
        selectedIndex = operatorIndex
        selectedGradient = gradient

    totalNorm = np.sqrt(totalNorm)
    
    print("Total gradient norm: {}".format(totalNorm))

    return selectedIndex, selectedGradient, totalNorm

  def calculateEnergy(self,coefficients,indices):
    '''
    Calculates the energy in a specified state.

    Arguments:
      coefficients (list): coefficients of the ansatz indices
      indices (list): indices that specify which operators are in the ansatz.

    Returns:
      energy (float): the energy in this state.
    '''

    assert(len(coefficients) == len(indices))
    
    ket = self.sparseReferenceState

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in 
    #the ansatz, following the order of the list
    for (coefficient,operatorIndex) in zip(coefficients,indices):

      # Multiply the operator by the respective coefficient
      sparseOperator = coefficient * self.sparsePool[operatorIndex]

      # Exponentiate the operator and update ket to represent the state after
      #this operator has been applied
      expOperator = scipy.sparse.linalg.expm(sparseOperator)
      ket = expOperator * ket

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = (bra * self.sparseHamiltonian * ket)[0,0].real

    return energy

  def callback(self, nfev, parameters, energy, stddev):
    '''
    Function to gather intermediate data from the optimization. Fills the 
    instance variable optEvolution with the data from each step in the 
    optimization process.
    '''

    self.optEvolution['nfev'].append(nfev)
    self.optEvolution['parameters'].append(parameters.copy())
    self.optEvolution['energy'].append(energy)
    self.optEvolution['stddev'].append(stddev)
        
  def optimizeCircuit(self,initialCoefficients,indices):
    '''
    Optimizes a certain ansatz, filling the instance variable optEvolution 
      with the optimization data.

    Arguments:
      initialCoefficients (list): the coefficients that define the starting 
        point.
      indices (list): the indices that identify the ansatz to be optimized
    '''
    
    # Initialize instance variable, that will be filled with the data from this 
    #optimization. The data from the previous one will be deleted.
    self.optEvolution = {
        'nfev': [],
        'parameters': [],
        'energy': [],
        'stddev': []
    }

    # Initialize parameter vector and circuit in Qiskit
    parameters = ParameterVector("Params",len(indices))
    ansatz = QuantumCircuit(self.qubitNumber)

    # Apply gates that prepare the reference determinant
    for i,state in enumerate(self.referenceDeterminant):
      if state == 1:
        ansatz.x(self.qubitNumber - 1 - i)

    # Apply gates corresponding to each of the ansatz operators.
    for i, op in enumerate(indices):
      pauliToCircuit(self.pool[op],
                      parameters[i],
                      ansatz,
                      self.qubitNumber)

    vqe = VQE(estimator=Estimator(),
              ansatz=ansatz,
              optimizer = COBYLA(rhobeg = 0.1), 
              callback = self.callback,
              initial_point = initialCoefficients)

    result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

  def run(self):
    '''
    Run the full Adapt VQE algorithm, until either the convergence condition is
      met or the maximum number of iterations is reached.
    '''

    while self.data.iterationCounter < self.maxIterations:
      print("==================SELF.DATA.RESULT: ", self.data.result)

      print("\n*** Adapt Iteration {} ***\n".format
            (self.data.iterationCounter + 1))

      maxIndex, maxGradient, totalNorm = self.selectOperator()
      maxOperator = self.pool[maxIndex]

      if totalNorm < self.threshold:

        print("\nConvergence condition achieved!\n")
        self.data.close(success = True)

        if self.data.result["energy"] is None:
          print("The chosen threshold was too large;" 
                " no iterations were completed.")
          return

        print(">>>>>>>>>>>>>>>Final Energy:", self.data.result["energy"])
        error = self.data.result["energy"] - molecule.fci_energy
        print("Error:",error)
        print("(in % of chemical accuracy: {:.3f}%)\n".format
              (error/chemicalAccuracy*100))
        
        print("Ansatz Indices:", self.data.result["indices"])
        print("Coefficients:", self.data.result["coefficients"])

        return

      print("Selected: {}, index {}".format(self.pool[maxIndex],maxIndex))

      print("(gradient: {})".format(maxGradient))

      newIndices = self.data.current["indices"].copy()
      newIndices.append(maxIndex)

      # Initialize the coefficient of the operator that will be newly added at 0
      newCoefficients = copy.deepcopy(self.data.current["coefficients"])
      newCoefficients.append(0)

      print("\nOptimizing energy with indices {}...".format
            (self.data.current["indices"] + [maxIndex]))
      indices = self.data.current["indices"] + [maxIndex]

      if self.backend is None:
        # Use sparse matrices
        opt_result = scipy.optimize.minimize(self.calculateEnergy, 
                                            newCoefficients,
                                            indices,
                                            method = "COBYLA",
                                            #tol = 10**-4,
                                            options = {'rhobeg': 0.1,
                                                        'disp': True})
        optimizedCoefficients = list(opt_result.x)
        nfev = opt_result.nfev

      else:
        # Optimize the ansatz in Qiskit
        self.optimizeCircuit(newCoefficients,indices)

        evolution = self.optEvolution
        optimizedCoefficients = list(self.optEvolution["parameters"][-1])
        nfev = self.optEvolution["nfev"][-1]
        
      optimizedEnergy = self.calculateEnergy(optimizedCoefficients,
                                             indices)
      
      print("Number of function evaluations:",nfev)
      print("Optimized energy: ",optimizedEnergy)
      print("Optimized coefficients: ", optimizedCoefficients)
        
      self.data.processIteration(maxIndex,
                             maxOperator,
                             optimizedEnergy,
                             totalNorm,
                             maxGradient,
                             optimizedCoefficients)
      
      print("Current ansatz:", self.data.current["ansatz"])
      
      # Update current state
      newState = self.computeState()
      self.data.current["state"] = newState
      
      print("\nEnergy Changes Associated with the Indices:",
            self.data.evolution["energyChange"])
      print("Performances Associated with the Indices: ",
            self.data.current["ansatz performances"])
      
    print("==================SELF.DATA.RESULT: ", self.data.result)
    
    print("\nThe maximum number of iterations ({}) was hit before the"
    " convergence criterion was satisfied.\n"
    "(current gradient norm is {} > {})"
    .format(self.maxIterations,self.data.current["total norm"],self.threshold))
    self.data.close(success = False)
