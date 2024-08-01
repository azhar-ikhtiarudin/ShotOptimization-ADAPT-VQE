from qiskit_aer import QasmSimulator
import copy
import numpy as np
import matplotlib.pyplot as plt
from src.molecular_def import h2 as molecule
chemicalAccuracy = 1.5936*10**-3

class AdaptData:
  '''
  Class meant to store data from an Adapt VQE run. 

  Methods:
    processIteration: to be called by the AdaptVQE class at the end of each 
      iteration
    close: to be called by the AdaptVQE class at the end of the run
    plot: to be called to plot data after the run 
  '''

  def __init__(self,
               initialEnergy,
               pool,
               sparsePool,
               referenceDeterminant,
               backend,
               shots,
               previousData = None):


    '''
    Initialize class instance
    Arguments:
      initialEnergy (float): energy of the reference state
      pool (list): operator pool
      sparsePool (list): sparse version of the pool
      referenceDeterminant (list): the Slater determinant to be used as reference,
        in big endian ordering (|abcd> <-> [a,b,c,d]; [qubit 0, qubit 1,...])
      backend (Union[None,qiskit.providers.ibmq.IBMQBackend]): the backend to 
        be used. If none, a simulation will be run using sparse matrices.
      shots (int): number of circuit repetitions
      previousData (AdaptData): data from a previous run, to be used as a 
        starting point
    ''' 


    self.pool = pool
    self.sparsePool = sparsePool
    self.backend = QasmSimulator()
    self.shots = shots
    
    if previousData is not None:
      # Starting point is the result of a previous Adapt run

      self.initialEnergy = initialEnergy
      self.evolution = copy.deepcopy(previousData.evolution)

      # Make sure the data corresponds to a complete run
      if not previousData.result:
        raise ValueError("Previous data does not supply final results.")

      self.current = copy.deepcopy(previousData.result)
      self.iterationCounter = previousData.iterationCounter

      # Make sure we're using the same pool
      # Comparing lenght is good enough
      if previousData is not None:
        assert(len(self.sparsePool) == len(previousData.sparsePool))

      '''
      assert(self.iterationCounter == len(self.evolution["selected gradient"]))
      assert(self.iterationCounter == len(self.evolution["total norm""]))
      assert(self.iterationCounter == len(self.evolution["coefficients"]))
      assert(self.iterationCounter == len(self.current["ansatz"]))
      assert(self.iterationCounter == len(self.current["coefficients"]))
      assert(self.iterationCounter == len(self.current["indices"]))
      '''

    else:
      # The initial energy is stored apart from the remaining ones, so that the 
      #index of the gradient norms in the beginning of each iteration 
      #corresponds to the index of the coefficients and energies in that 
      #iteration in the respective arrays.
      self.initialEnergy = initialEnergy

      self.evolution = {}
      self.evolution["energies"] = []
      self.evolution["total norm"] = []
      self.evolution["selected gradient"] = []
      self.evolution["coefficients"] = []
      self.evolution["energyChange"] = []
      self.evolution["indices"] = []

      self.current = {}
      self.current["ansatz"] = []
      self.current["coefficients"] = []
      self.current["indices"] = []
      # performance of a selected operator = | energy change / gradient | 
      self.current["ansatz performances"] = []
      self.current["performances"] = []
      self.current["energy"] = None
      self.current["state"] = None
      self.current["total norm"] = None

      self.iterationCounter = 0

    self.closed = False
    self.success = False

    assert(self.iterationCounter == len(self.evolution["selected gradient"]))
    assert(self.iterationCounter == len(self.evolution["total norm"]))
    assert(self.iterationCounter == len(self.evolution["coefficients"]))
 
    self.result = {}

  def processIteration(self,
                   operatorIndex,
                   operator,
                   energy,
                   totalNorm,
                   selectedGradient,
                   coefficients):
    '''
    Receives and processes the values fed to it by an instance of the AdaptVQE 
    class at the end of each run.

    Arguments:
      operatorIndex (int): index of the selected operator
      operator (union[openfermion.QubitOperator, openfermion.FermionOperator]):
        the selected operator
      energy (float): the optimized energy, at the end of the iteration
      totalNorm (int): the norm of the total gradient norm at the beggining 
        of this iteration
      selectedGradient (float): the absolute value of the gradient of the 
        operator that was added in this iteration
      coefficients (list): a list of the coefficients selected by the optimizer
        in this iteration
    '''

    if not isinstance(energy,float):
      raise TypeError("Expected float, not {}.".format(type(energy).__name__))
    if not isinstance(totalNorm,float):
      raise TypeError("Expected float, not {}.".format(type(totalNorm).__name__))
    if not isinstance(selectedGradient,float):
      raise TypeError("Expected float, not {}.".format(type(selectedGradient).__name__))
    if not isinstance(coefficients,list):
      raise TypeError("Expected list, not {}.".format(type(coefficients).__name__))

    if len(coefficients) != len(self.current["ansatz"]) + 1:
      raise ValueError("The length of the coefficient list should match the"
      " ansatz size ({} != {}).".format
        (len(coefficients),len(self.current["ansatz"]) + 1))

    if totalNorm < 0:
      raise ValueError("Total gradient norm should be positive; its {}".\
                       format(totalNorm))
      
    self.current["coefficients"] = copy.deepcopy(coefficients)
    
    if self.iterationCounter == 0:
      previousEnergy = self.initialEnergy
    else:
      previousEnergy = self.current["energy"] 

    energyChange = energy - previousEnergy
    performance = np.abs(energyChange / selectedGradient)

    self.current["ansatz performances"].append(performance)
    self.current["performances"].append(performance)

    ansatzPerformance = np.average(self.current["ansatz performances"])

    print("Energy Change: ",energyChange)
    print("Performance ratio: ",performance)
    print("\nCurrent average performance ratio: ", 
          np.average(self.current["performances"]))
    print("Current 10-last average performance ratio: ", 
          np.average(self.current["performances"][-10:]))
    print("Current average performance ratio of the ansatz: ",
          ansatzPerformance)

    self.current["ansatz"].append(operator)
    self.current["energy"] = energy
    self.current["indices"].append(operatorIndex)
    self.current["total norm"] = totalNorm

    coefficientCopy = copy.deepcopy(self.current["coefficients"])
    indicesCopy = copy.deepcopy(self.current["indices"])

    self.evolution["energies"].append(energy)
    self.evolution["energyChange"].append(energyChange)
    self.evolution["total norm"].append(totalNorm)
    self.evolution["selected gradient"].append(selectedGradient)
    self.evolution["indices"].append(indicesCopy) 

    self.evolution["coefficients"].append(coefficientCopy)

    self.iterationCounter += 1

    #assert(self.iterationCounter == len(self.evolution["total norm"]))
    #assert(self.iterationCounter == len(self.evolution["coefficients"]))
    #assert(self.iterationCounter == len(self.evolution["energies"]))

  def close(self,success):
    '''
    To be called at the end of the run, to close the data structures

    Arguments:
      success (bool): True if the convergence condition was met, False if not
        (the maximum number of iterations was met before that)
    '''
    print("CLOSE")
    print("sel.current: ", self.current)
    self.result = self.current
    self.closed = True
    self.success = success

  def plot(self, plotGradient = True, detailedTitle = True):
    '''
    Plots the evolution of the energy along the run.

    Arguments:
      plotGradient (bool): whether the total gradient norm should be plotted 
        as well.
      detailedTitle (bool): whether the title should have include which was the
        used backend and the shot number.
    '''

    iterationNumber = self.iterationCounter
    iterationLabels = [iterationLable 
                      for iterationLable in range(0,self.iterationCounter+1)]


    gradientNorms = self.evolution["total norm"]
    energies = [self.initialEnergy] + self.evolution["energies"]

    fig, ax1 = plt.subplots(figsize=[8,8])
    title = "Qubit Adapt VQE for {}".format(molecule.description)

    # if self.backend is None:
    #   backendName = None

    # elif self.backend.name() == "statevector_simulator":
    #   backendName = "Aer State Vector Simulator"

    # elif self.backend.name() == "qasm_simulator":
    #   backendName = "Aer QASM Simulator"

    # else:
    #   backendName = self.backend.name()

    # if detailedTitle:
    #   title += "\nr = {}Å".format(str(bondLength))
    #   if backendName is not None:
    #     title += ", {}".format(backendName)
    #   if backendName != "Aer State Vector Simulator":
    #     title += ", {} shots".format(self.shots)

    # plt.title(title)

    color1 = 'b'  
    ax1.plot(iterationLabels, energies, '--o',color = color1)
    ax1.tick_params(axis='y', labelcolor = color1)

    # Shade area within chemical accuracy
    exactEnergy = molecule.fci_energy
    minAccuracy = exactEnergy - chemicalAccuracy
    maxAccuracy = exactEnergy + chemicalAccuracy
    l = ax1.axhspan(minAccuracy, maxAccuracy, alpha=0.3, color = 'cornflowerblue')

    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Energy (au)",color = color1)
    
    plt.xticks(range(1, 1 + self.iterationCounter))

    if plotGradient:
      color2 = 'r'
      ax2 = ax1.twinx()
      ax2.plot(iterationLabels[1:], gradientNorms, '--o',color = color2)
      ax2.tick_params(axis='y', labelcolor = color2)
      ax2.set_ylabel("Total Gradient Norm",color = color2)