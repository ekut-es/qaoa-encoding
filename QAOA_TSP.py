import qiskit as qk
import numpy as np
from scipy.optimize import minimize

from abc import ABC, abstractmethod
from itertools import product

class QAOA_TSP(ABC):

    def compute_path_length(path : tuple, adj_matrix : np.ndarray) -> float:
        length = adj_matrix[path[-1], path[0]]
        for i, j in zip(path[:-1], path[1:]):
            length += adj_matrix[i, j]
        return length

    def __init__(self):
        # Qubit and classical bit registers
        self.qubits = None
        self.cbits = None
        # Circuit parameters
        self.beta = None
        self.gamma = None
        self.matrix = None
        # Circuits at different stages of parameter binding
        self.circuit = None
        self.matrix_bound = None
        self.params_bound = None
        self.backend = None
        # Solving variables
        self.adj_matrix = None
        self.number_of_iterations = 0

    @abstractmethod
    def get_number_of_qubits(self) -> int:
        pass

    @abstractmethod
    def build_state_preparation(self):
        pass

    @abstractmethod
    def build_phase_separator(self, gamma : qk.circuit.Parameter):
        pass

    @abstractmethod
    def build_mixer(self, beta : qk.circuit.Parameter):
        pass

    @abstractmethod
    def compute_expectation(self, counts : dict, shots : int) -> float:
        pass

    @abstractmethod
    def path_from_string(string : str) -> tuple:
        pass

    def build_circuit(self, n : int, p : int, backend : qk.providers.Backend):
        self.matrix_bound = None
        self.params_bound = None

        self.n = n
        self.p = p
        self.backend = backend

        self.beta = [qk.circuit.Parameter("beta{}".format(i)) for i in range(p)]
        self.gamma = [qk.circuit.Parameter("gamma{}".format(i)) for i in range(p)]
        self.matrix = [[qk.circuit.Parameter("matrix{0}{1}".format(i,j)) for j in range(n)] for i in range(n)]

        self.number_of_qubits = self.get_number_of_qubits()

        self.qubits = qk.QuantumRegister(self.number_of_qubits)
        self.cbits = qk.ClassicalRegister(self.number_of_qubits)

        self.circuit = qk.QuantumCircuit(self.qubits, self.cbits)

        self.build_state_preparation()
        # self.build_mixer(self.beta[0])

        for i in range(self.p):
            self.build_phase_separator(self.gamma[i])
            # self.build_mixer(self.beta[i+1])
            self.build_mixer(self.beta[i])

        self.circuit.measure(self.qubits, self.cbits)
        self.circuit = qk.transpile(self.circuit, optimization_level=3, backend=backend)

    def bind_matrix(self, adj_matrix : np.ndarray):
        assert (self.circuit is not None), "Circuit needs to be built first"
        adj_matrix_norm = adj_matrix/(np.max(adj_matrix)*self.n)
        params_matrix = {self.matrix[i][j]: adj_matrix_norm[i,j] for i, j in product(range(self.n), range(self.n)) if i != j}
        self.matrix_bound = self.circuit.assign_parameters(params_matrix)
        self.adj_matrix = adj_matrix

    def bind_parameters(self, parameters : list):
        assert (self.matrix_bound is not None), "Matrix parameters need to be bound"
        betas = parameters[:self.p]
        gammas = parameters[self.p:]

        params_beta = {qcbeta: pbeta for qcbeta, pbeta in zip(self.beta, betas)}
        params_gamma = {qcgamma: pgamma for qcgamma, pgamma in zip(self.gamma, gammas)}

        self.params_bound = self.matrix_bound.assign_parameters({**params_beta, **params_gamma})

    def run_circuit(self, shots : int) -> dict:
        assert (self.params_bound is not None), "Circuit parameters need to be bound"
        counts = self.backend.run(self.params_bound, seed_simulator=42, shots=shots).result().get_counts()
        # Reverse all measured strings because qiskit decided that was a good idea
        return {string[::-1]: count for string, count in counts.items()}
    
    def solve(self, adj_matrix : np.ndarray, shots : int = 1000):

        self.bind_matrix(adj_matrix)

        def get_circuit_expectation(parameters):
            self.bind_parameters(parameters)
            counts = self.run_circuit(shots)
            expectation = self.compute_expectation(counts, shots)
            self.number_of_iterations += 1
            return expectation
        
        self.number_of_iterations = 0
        res = minimize(get_circuit_expectation, [1.0]*(self.p*2), method='COBYLA')
        optim = res.x

        self.bind_parameters(optim)
        counts = self.run_circuit(shots)
        best_path = max(counts, key=counts.get)
        best_path = self.path_from_string(best_path)

        return best_path, QAOA_TSP.compute_path_length(best_path, adj_matrix)