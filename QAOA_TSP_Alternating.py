import qiskit as qk
import numpy as np
from itertools import product, permutations

from QAOA_TSP import QAOA_TSP

def Swap4Gate(theta):
    gate = qk.QuantumCircuit(4)
    gate.cx(1, 0)
    gate.ch(0, 1)
    gate.cx(2, 3)
    gate.ch(3, 2)
    gate.cx(1, 2)
    gate.append(qk.circuit.library.standard_gates.RZGate(theta).control(2), (0, 3, 2))
    gate.cx(1, 2)
    gate.ch(3, 2)
    gate.cx(2, 3)
    gate.ch(0, 1)
    gate.cx(1, 0)
    return gate.to_gate()

class QAOA_TSP_Alternating(QAOA_TSP):

    def get_commutative_mapping(n_vertices : int):
        # TODO: Solve edge coloring problem
        # Return static solution for n_vertices <= 5
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
        elif n_vertices == 5:
            p_col = (frozenset((frozenset((0,1)), frozenset((2,4)))), frozenset((frozenset((0,2)), frozenset((3,4)))), frozenset((frozenset((0,3)), frozenset((1,2)))),
                    frozenset((frozenset((0,4)), frozenset((1,3)))), frozenset((frozenset((1,4)), frozenset((2,3)))))
            p_par = (frozenset((0,2)), frozenset((1,3)), frozenset((4,)))

            return list(product(p_par, p_col))
        else:
            raise Exception("Works only for n_vertices <= 5 in this version")
        
    def node_timestep_to_index(self, node : int, timestep : int) -> int:
        node = node % self.n
        timestep = timestep % self.n
        return node*self.n + timestep
        
    def get_number_of_qubits(self) -> int:
        return self.n**2
    
    def compute_string_cost(self, string : str, adj_matrix : np.ndarray) -> float:
        cost = 0
        for layer in self.mapping:
            for timestep, qubits in product(layer[0], layer[1]):
                qs = tuple(qubits)
                id1 = self.node_timestep_to_index(qs[0], timestep)
                id2 = self.node_timestep_to_index(qs[1], timestep + 1)
                if string[id1] == string[id2]:
                    cost += adj_matrix[qs[0]][qs[1]]
                else:
                    cost -= adj_matrix[qs[0]][qs[1]]

                id1 = self.node_timestep_to_index(qs[1], timestep)
                id2 = self.node_timestep_to_index(qs[0], timestep + 1)
                if string[id1] == string[id2]:
                    cost += adj_matrix[qs[1]][qs[0]]
                else:
                    cost -= adj_matrix[qs[1]][qs[0]]

        return cost
    
    def compute_path_lengths(self, adj_matrix : np.ndarray):
        # This is kind of cheating but the Hamiltonian is too large to store
        self.path_lengths = {}
        for perm in permutations(range(self.n)):
            string = ""
            for i in range(self.n):
                k = perm.index(i)
                string += "0"*k + "1" + "0"*(self.n-k-1)
            
            self.path_lengths[string] = self.compute_string_cost(string, adj_matrix)
    
    def build_state_preparation(self):
        for i in range(0, self.number_of_qubits, self.n + 1):
            self.circuit.x(i)

    def build_phase_separator(self, gamma : qk.circuit.Parameter):
        for layer in self.mapping:
            # This should all happen in depth 1
            for timestep, qubits in product(layer[0], layer[1]):
                qs = tuple(qubits)

                id1 = self.node_timestep_to_index(qs[0], timestep)
                id2 = self.node_timestep_to_index(qs[1], timestep + 1)
                self.circuit.rzz(2 * gamma * self.matrix[qs[0]][qs[1]], id1, id2)

                id1 = self.node_timestep_to_index(qs[1], timestep)
                id2 = self.node_timestep_to_index(qs[0], timestep + 1)
                self.circuit.rzz(2 * gamma * self.matrix[qs[1]][qs[0]], id1, id2)

    def build_mixer(self, beta : qk.circuit.Parameter):
        def four_qubit_swap(u, v, t):
            i = t
            ip1 = t+1
            ui = self.node_timestep_to_index(u, i)
            uip1 = self.node_timestep_to_index(u, ip1)
            vi = self.node_timestep_to_index(v, i)
            vip1 = self.node_timestep_to_index(v, ip1)
            self.circuit.append(swap_gate, (ui, vi, uip1, vip1))

        swap_gate = Swap4Gate(beta)
        for layer in self.mapping:
            # This should all happen in depth 1
            for timestep, qubits in product(layer[0], layer[1]):
                qs = tuple(qubits)
                four_qubit_swap(qs[0], qs[1], timestep)

    def compute_expectation(self, counts : dict, shots : int) -> float:
        sum_count = 0
        for string, count in counts.items():
            sum_count += self.path_lengths[string]*count

        return sum_count/shots
    
    def path_from_string(self, string : str) -> tuple:
        path = [-1]*self.n
        for i in range(self.n):
            node_string = string[i*self.n:i*self.n+self.n]
            node_position = node_string.find('1')
            path[node_position] = i
        # path.append(0)
        return tuple(path)

    def build_circuit(self, n : int, p : int, backend : qk.providers.Backend):
        self.mapping = QAOA_TSP_Alternating.get_commutative_mapping(n)
        super().build_circuit(n, p, backend)

    def bind_matrix(self, adj_matrix : np.ndarray):
        super().bind_matrix(adj_matrix)
        self.compute_path_lengths(adj_matrix)
