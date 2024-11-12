

from copy import deepcopy

class AnsatzData:

    def __init__(self, coefficients=[], indices=[], sel_gradients=[]):
        self.coefficients = coefficients
        self.indices = indices
        self.sel_gradients = sel_gradients

    def grow(self, indices, new_coefficients, sel_gradients):
        self.indices = indices
        self.coefficients = new_coefficients
        self.sel_gradients = np.append(self.sel_gradients, sel_gradients)

    def remove(self, index, new_coefficients):
        self.indices.pop(index)
        self.coefficients = new_coefficients
        rem_grad = self.sel_gradients.pop(index)

        return rem_grad

    @property
    def size(self):
        return len(self.indices)


class IterationData:
    def __init__(self,
                 ansatz=None,
                 energy=None,
                 error=None,
                 energy_change=None,
                 gradient_norm=None,
                 sel_gradients=None,
                 inv_hessian=None,
                 gradients=None,
                 nfevs=None,
                 ngevs=None,
                 nits=None
                 ):
        if ansatz:
            self.ansatz = deepcopy(ansatz)
        else:
            self.ansatz = AnsatzData()
        
        self.energy = energy
        self.energy_change = energy_change
        self.error = error
        self.gradient_norm = gradient_norm
        self.sel_gradients = sel_gradients
        self.inv_hessian = inv_hessian
        self.gradients = gradients
        self.nfevs = nfevs
        self.ngevs = ngevs
        self.nits = nits

class EvolutionData:

    def __init__(self, initial_energy, prev_ev_data=None):

        self.initial_energy = initial_energy

        if prev_ev_data:
            self.its_data = prev_ev_data.its_data
        else:
            # List of IterationData objects
            self.its_data = []

    def reg_it(
        self,
        coefficients,
        indices,
        energy,
        error,
        gradient_norm,
        sel_gradients,
        inv_hessian,
        gradients,
        nfevs,
        ngevs,
        nits,
    ):

        if self.its_data:
            previous_energy = self.last_it.energy
        else:
            previous_energy = self.initial_energy

        energy_change = energy - previous_energy

        ansatz = deepcopy(self.last_it.ansatz)
        ansatz.grow(indices, coefficients, sel_gradients)

        it_data = IterationData(
            ansatz,
            energy,
            error,
            energy_change,
            gradient_norm,
            sel_gradients,
            inv_hessian,
            gradients,
            nfevs,
            ngevs,
            nits,
        )

        self.its_data.append(it_data)

        return

    @property
    def coefficients(self):
        return [it_data.ansatz.coefficients for it_data in self.its_data]

    @property
    def energies(self):
        return [it_data.energy for it_data in self.its_data]

    @property
    def inv_hessians(self):
        return [it_data.inv_hessian for it_data in self.its_data]

    @property
    def gradients(self):
        return [it_data.gradients for it_data in self.its_data]

    @property
    def errors(self):
        return [it_data.error for it_data in self.its_data]

    @property
    def energy_changes(self):
        return [it_data.energy_change for it_data in self.its_data]

    @property
    def gradient_norms(self):
        return [it_data.gradient_norm for it_data in self.its_data]

    @property
    def indices(self):
        return [it_data.ansatz.indices for it_data in self.its_data]

    @property
    def nfevs(self):
        return [it_data.nfevs for it_data in self.its_data]

    @property
    def ngevs(self):
        return [it_data.ngevs for it_data in self.its_data]

    @property
    def nits(self):
        return [it_data.nits for it_data in self.its_data]

    @property
    def sel_gradients(self):
        return [it_data.sel_gradients for it_data in self.its_data]

    @property
    def sizes(self):
        return [len(it_data.ansatz.indices) for it_data in self.its_data]

    @property
    def last_it(self):

        if self.its_data:
            return self.its_data[-1]
        else:
            # No data yet. Return empty IterationData object
            return IterationData()


class AdaptData:
    def __init__(
            self, initial_energy, pool, sparse_ref_state, file_name, fci_energy, n
    ):
        self.pool_name = pool.name
        self.initial_energy = initial_energy
        self.initial_error = initial_energy - fci_energy
        self.sparse_ref_state = sparse_ref_state

        self.evolution = EvolutionData(initial_energy)
        self.file_name = file_name
        self.iteration_counter = 0
        self.fci_energy = fci_energy
        self.n = n

        self.closed = False
        self.success= False

    def process_iteration(
            self,
            indices,
            energy,
            gradient_norm,
            selected_gradients,
            coefficients,
            inv_hessian,
            gradients,
            nfevs,
            ngevs,
            nits
    ):
        error = energy - self.fci_energy
        self.evolution.reg_it(
            coefficients,
            indices,
            energy,
            error,
            gradient_norm,
            selected_gradients,
            inv_hessian,
            gradients,
            nfevs,
            ngevs,
            nits,
        )

        self.iteration_counter += 1

        return energy
    
    def close(self, success, file_name = None):
        self.result = self.evolution.last_it
        self.closed = True
        self.success = success
        if file_name is not None:
            self.file_name = file_name
    
    @property
    def current(self):
        if self.evolution.its_data:
            return self.evolution.last_it
        else:
            return IterationData(energy=self.initial_energy)