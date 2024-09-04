import qiskit as qk
import numpy as np
from scipy.optimize import minimize
from itertools import product, permutations

def qubit_timestep_to_index(qubit, timestep, n):
    qubit = qubit % n
    timestep = timestep % n
    return qubit*n + timestep

def path_from_string_orig(string, amount_nodes):
    path = [-1]*amount_nodes
    for i in range(amount_nodes):
        node_string = string[i*amount_nodes:i*amount_nodes+amount_nodes]
        node_position = node_string.find('1')
        path[node_position] = i
    return path

def compute_path_length(path, adj_matrix):
    length = 0
    for i,j in zip(path[:-1], path[1:]):
        length += adj_matrix[i,j]
    return length

def compute_string_cost(string, adj_matrix, mapping, n):
    cost = 0
    for layer in mapping:
        for timestep, qubits in product(layer[0], layer[1]):
            qs = tuple(qubits)
            id1 = qubit_timestep_to_index(qs[0], timestep, n)
            id2 = qubit_timestep_to_index(qs[1], timestep+1, n)
            if string[id1] == string[id2]:
                cost += adj_matrix[qs[0]][qs[1]]
            else:
                cost -= adj_matrix[qs[0]][qs[1]]

            id1 = qubit_timestep_to_index(qs[1], timestep, n)
            id2 = qubit_timestep_to_index(qs[0], timestep+1, n)
            if string[id1] == string[id2]:
                cost += adj_matrix[qs[0]][qs[1]]
            else:
                cost -= adj_matrix[qs[0]][qs[1]]

    return cost
    

def get_commutative_mapping(n_vertices):
    # TODO: Solve edge coloring problem
    # Return static solution for n_vertices <= 4
    if n_vertices == 3:
        p_col = (frozenset((frozenset((0,1)),)), frozenset((frozenset((0,2)),)),
                 frozenset((frozenset((1,2)),)))
        p_par = (frozenset((0,)), frozenset((1,)), frozenset((2,)))
        
        return list(product(p_par, p_col))
    elif n_vertices == 4:
        p_col = (frozenset((frozenset((0,1)), frozenset((2,3)))), frozenset((frozenset((0,2)), frozenset((1,3)))),
                 frozenset((frozenset((0,3)), frozenset((1,2)))))
        p_par = (frozenset((0,2)), frozenset((1,3)))
        
        return list(product(p_par, p_col))
    else:
        # TODO: Solve edge coloring problem
        raise Exception("Works only for n_vertices <= 4 in this version")
        
class TSP_QAOA_Orig:
    def __init__(self, n, p):
        assert (n > 2) and (n < 5)
        
        self.circuit = None
        self.matrix_bound = None
        self.params_bound = None
        self.beta = None
        self.gamma = None
        self.matrix = None
        self.path_lengths = None
        self.p = p
        self.n = n
        self.mapping = get_commutative_mapping(n)
        self.opt_iterations = 0

    def build_state_preparation(self):
        if self.n == 3:
            self.circuit.ry(np.arccos(-1/3), 0)
            self.circuit.ch(0, 1)
            self.circuit.cx(1, 2)
            self.circuit.cx(0, 1)
            self.circuit.x(0)

            self.circuit.x(6)
            self.circuit.x(7)
            self.circuit.x(8)

            self.circuit.cx(0, 6)
            self.circuit.cx(1, 7)
            self.circuit.cx(2, 8)

            # cch
            self.circuit.ry(np.pi/4, 3)
            self.circuit.ccx(6, 7, 3)
            self.circuit.ry(-np.pi/4, 3)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (3, 6, 7, 4))
            self.circuit.ccx(6, 7, 3)

            self.circuit.ry(np.pi/4, 4)
            self.circuit.ccx(7, 8, 4)
            self.circuit.ry(-np.pi/4, 4)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (4, 7, 8, 5))
            self.circuit.ccx(7, 8, 4)

            self.circuit.ry(np.pi/4, 3)
            self.circuit.ccx(6, 8, 3)
            self.circuit.ry(-np.pi/4, 3)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (3, 6, 8, 5))
            self.circuit.ccx(6, 8, 3)

            self.circuit.cx(3, 6)
            self.circuit.cx(4, 7)
            self.circuit.cx(5, 8)
        elif self.n == 4:
            self.circuit.ry(np.arccos(-1/2), 0)
            self.circuit.cry(np.arccos(-1/3), 0, 1)
            self.circuit.ch(1, 2)
            self.circuit.cx(2, 3)
            self.circuit.cx(1, 2)
            self.circuit.cx(0, 1)
            self.circuit.x(0)

            self.circuit.x(12)
            self.circuit.x(13)
            self.circuit.x(14)
            self.circuit.x(15)

            self.circuit.cx(0, 12)
            self.circuit.cx(1, 13)
            self.circuit.cx(2, 14)
            self.circuit.cx(3, 15)

            self.circuit.ry(np.arccos(-1/3), 9)
            self.circuit.ch(9, 10)
            self.circuit.cx(10, 11)
            self.circuit.cx(9, 10)
            self.circuit.x(9)

            self.circuit.cswap(12, 4, 11)
            self.circuit.swap(10, 11)
            self.circuit.swap(9, 11)
            self.circuit.cswap(13, 5, 11)
            self.circuit.cswap(13, 10, 11)
            self.circuit.swap(9, 11)
            self.circuit.cswap(14, 6, 11)
            self.circuit.cswap(14, 10, 11)
            self.circuit.cswap(14, 9, 11)
            self.circuit.cswap(15, 7, 11)

            self.circuit.cx(4, 12)
            self.circuit.cx(5, 13)
            self.circuit.cx(6, 14)
            self.circuit.cx(7, 15)

            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 13, 8)
            self.circuit.ry(-np.pi/4, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 13, 9))
            self.circuit.ccx(12, 13, 8)

            self.circuit.ry(np.pi/4, 9)
            self.circuit.ccx(13, 14, 9)
            self.circuit.ry(-np.pi/4, 9)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (9, 13, 14, 10))
            self.circuit.ccx(13, 14, 9)

            self.circuit.ry(np.pi/4, 10)
            self.circuit.ccx(14, 15, 10)
            self.circuit.ry(-np.pi/4, 10)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (10, 14, 15, 11))
            self.circuit.ccx(14, 15, 10)

            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 14, 8)
            self.circuit.ry(-np.pi/4, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 14, 10))
            self.circuit.ccx(12, 14, 8)

            self.circuit.ry(np.pi/4, 9)
            self.circuit.ccx(13, 15, 9)
            self.circuit.ry(-np.pi/4, 9)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (9, 13, 15, 11))
            self.circuit.ccx(13, 15, 9)

            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 15, 8)
            self.circuit.ry(-np.pi/4, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 15, 11))
            self.circuit.ccx(12, 15, 8)

            self.circuit.cx(8, 12)
            self.circuit.cx(9, 13)
            self.circuit.cx(10, 14)
            self.circuit.cx(11, 15)
        else:
            pass

    def reverse_state_preparation(self):
        if self.n == 3:
            self.circuit.cx(5, 8)
            self.circuit.cx(4, 7)
            self.circuit.cx(3, 6)

            self.circuit.ccx(6, 8, 3)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (3, 6, 8, 5))
            self.circuit.ry(np.pi/4, 3)
            self.circuit.ccx(6, 8, 3)
            self.circuit.ry(-np.pi/4, 3)

            self.circuit.ccx(7, 8, 4)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (4, 7, 8, 5))
            self.circuit.ry(np.pi/4, 4)
            self.circuit.ccx(7, 8, 4)
            self.circuit.ry(-np.pi/4, 4)

            self.circuit.ccx(6, 7, 3)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (3, 6, 7, 4))
            self.circuit.ry(np.pi/4, 3)
            self.circuit.ccx(6, 7, 3)
            self.circuit.ry(-np.pi/4, 3)

            self.circuit.cx(2, 8)
            self.circuit.cx(1, 7)
            self.circuit.cx(0, 6)

            self.circuit.x(8)
            self.circuit.x(7)
            self.circuit.x(6)

            self.circuit.x(0)
            self.circuit.cx(0, 1)
            self.circuit.cx(1, 2)
            self.circuit.ch(0, 1)
            self.circuit.ry(-np.arccos(-1/3), 0)
        elif self.n == 4:
            self.circuit.cx(11, 15)
            self.circuit.cx(10, 14)
            self.circuit.cx(9, 13)
            self.circuit.cx(8, 12)

            self.circuit.ccx(12, 15, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 15, 11))
            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 15, 8)
            self.circuit.ry(-np.pi/4, 8)

            self.circuit.ccx(13, 15, 9)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (9, 13, 15, 11))
            self.circuit.ry(np.pi/4, 9)
            self.circuit.ccx(13, 15, 9)
            self.circuit.ry(-np.pi/4, 9)

            self.circuit.ccx(12, 14, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 14, 10))
            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 14, 8)
            self.circuit.ry(-np.pi/4, 8)

            self.circuit.ccx(14, 15, 10)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (10, 14, 15, 11))
            self.circuit.ry(np.pi/4, 10)
            self.circuit.ccx(14, 15, 10)
            self.circuit.ry(-np.pi/4, 10)

            self.circuit.ccx(13, 14, 9)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (9, 13, 14, 10))
            self.circuit.ry(np.pi/4, 9)
            self.circuit.ccx(13, 14, 9)
            self.circuit.ry(-np.pi/4, 9)

            self.circuit.ccx(12, 13, 8)
            self.circuit.append(qk.circuit.library.standard_gates.C3XGate(), (8, 12, 13, 9))
            self.circuit.ry(np.pi/4, 8)
            self.circuit.ccx(12, 13, 8)
            self.circuit.ry(-np.pi/4, 8)

            self.circuit.cx(7, 15)
            self.circuit.cx(6, 14)
            self.circuit.cx(5, 13)
            self.circuit.cx(4, 12)

            self.circuit.cswap(15, 7, 11)
            self.circuit.cswap(14, 9, 11)
            self.circuit.cswap(14, 10, 11)
            self.circuit.cswap(14, 6, 11)
            self.circuit.swap(9, 11)
            self.circuit.cswap(13, 10, 11)
            self.circuit.cswap(13, 5, 11)
            self.circuit.swap(9, 11)
            self.circuit.swap(10, 11)
            self.circuit.cswap(12, 4, 11)

            self.circuit.x(9)
            self.circuit.cx(9, 10)
            self.circuit.cx(10, 11)
            self.circuit.ch(9, 10)
            self.circuit.ry(-np.arccos(-1/3), 9)

            self.circuit.cx(3, 15)
            self.circuit.cx(2, 14)
            self.circuit.cx(1, 13)
            self.circuit.cx(0, 12)

            self.circuit.x(15)
            self.circuit.x(14)
            self.circuit.x(13)
            self.circuit.x(12)

            self.circuit.x(0)
            self.circuit.cx(0, 1)
            self.circuit.cx(1, 2)
            self.circuit.cx(2, 3)
            self.circuit.ch(1, 2)
            self.circuit.cry(-np.arccos(-1/3), 0, 1)
            self.circuit.ry(-np.arccos(-1/2), 0)
        else:
            pass

    def build_phase_separator(self, parameter):
        """
        Phase separator for a single iteration, hence only a single parameter is used
        """
        for layer in self.mapping:
            # This should all happen in depth 1
            for timestep, qubits in product(layer[0], layer[1]):
                qs = tuple(qubits)

                id1 = qubit_timestep_to_index(qs[0], timestep, self.n)
                id2 = qubit_timestep_to_index(qs[1], timestep+1, self.n)
                self.circuit.rzz(2 * parameter * self.matrix[qs[0]][qs[1]], id1, id2)

                id1 = qubit_timestep_to_index(qs[1], timestep, self.n)
                id2 = qubit_timestep_to_index(qs[0], timestep+1, self.n)
                self.circuit.rzz(2 * parameter * self.matrix[qs[1]][qs[0]], id1, id2)


    def build_mixer(self, parameter):
        """
        Mixer for a single iteration, hence only a single parameter is used
        """
        self.reverse_state_preparation()

        self.circuit.x(self.qubits)

        cp_gate = qk.circuit.library.standard_gates.PhaseGate(parameter).control(len(self.qubits) - 1)
        self.circuit.append(cp_gate, self.qubits)

        self.circuit.x(self.qubits)

        self.build_state_preparation()


    def compute_path_lengths(self, adj_matrix):
        # This is kind of cheating but the Hamiltonian is too large to store
        self.path_lengths = {}
        for perm in permutations(range(self.n)):
            string = ""
            for i in range(self.n):
                k = perm.index(i)
                string += "0"*k + "1" + "0"*(self.n-k-1)
            
            self.path_lengths[string] = compute_string_cost(string, adj_matrix, self.mapping, self.n)


    def compute_expectation(self, counts, shots):
        sum_count = 0
        for string, count in counts.items():
            sum_count += self.path_lengths[string]*count

        return sum_count/shots
       

    def build_circuit(self, backend=None):
        self.beta = [qk.circuit.Parameter("beta{}".format(i)) for i in range(self.p)]
        self.gamma = [qk.circuit.Parameter("gamma{}".format(i)) for i in range(self.p)]
        self.matrix = [[qk.circuit.Parameter("matrix{0}{1}".format(i,j)) for j in range(self.n)] for i in range(self.n)]

        self.qubits = qk.QuantumRegister(self.n**2)
        self.cbits = qk.ClassicalRegister(self.n**2)

        self.circuit = qk.QuantumCircuit(self.qubits, self.cbits)
        
        self.build_state_preparation()
        
        for i in range(self.p):
            self.build_phase_separator(self.gamma[i])
            self.build_mixer(self.beta[i])

        self.circuit.measure(self.qubits, self.cbits)
        self.circuit = qk.transpile(self.circuit, optimization_level=3, backend=backend)
    
    
    def bind_parameters(self, parameters):
        assert (self.matrix_bound is not None), "Matrix parameters need to be bound"
        betas = parameters[:self.p]
        gammas = parameters[self.p:]
        
        params_beta = {qcbeta: pbeta for qcbeta, pbeta in zip(self.beta, betas)}
        params_gamma = {qcgamma: pgamma for qcgamma, pgamma in zip(self.gamma, gammas)}
        
        self.params_bound = self.matrix_bound.assign_parameters({**params_beta, **params_gamma})
    

    def run_circuit(self, parameters, backend, shots=1000):
        self.bind_parameters(parameters)
        counts = backend.run(self.params_bound, seed_simulator=42, shots=shots).result().get_counts()
        return counts


    def bind_matrix(self, adj_matrix):
        adj_matrix_norm = adj_matrix/(np.max(adj_matrix)*self.n)
        params_matrix = {self.matrix[i][j]: adj_matrix_norm[i,j] for i, j in product(range(self.n), range(self.n)) if i != j}
        self.matrix_bound = self.circuit.assign_parameters(params_matrix)
        self.compute_path_lengths(adj_matrix_norm)

    
    def solve(self, adj_matrix, shots=1000, backend=None):
        assert (self.n == len(adj_matrix)), "Adjacency matrix does not fit with given number of qubits"
        
        if self.circuit is None:
            self.build_circuit(backend)

        self.bind_matrix(adj_matrix)
        
        if backend is None:
            backend = qk.Aer.get_backend('qasm_simulator')
        
        def get_circuit_expectation(parameters, evaluate=False):
            counts = self.run_circuit(parameters, backend, shots)
            expectation = self.compute_expectation(counts, shots)
            self.opt_iterations += 1
            return expectation
        
        self.opt_iterations = 0
        res = minimize(get_circuit_expectation, [1.0]*(self.p*2), method='COBYLA')
        optim = res.x
    
        counts = self.run_circuit(optim, backend)
        best_path = max(counts, key=counts.get)
        best_path = path_from_string_orig(best_path, self.n)
        best_path = best_path + [best_path[0]]
        path_length = compute_path_length(best_path, adj_matrix)
    
        return tuple(best_path[1:]), path_length