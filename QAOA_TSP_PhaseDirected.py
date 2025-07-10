import qiskit as qk
import numpy as np
from itertools import permutations

from QAOA_TSP import QAOA_TSP

class QAOA_TSP_PhaseDirected(QAOA_TSP):

    def build_state_preparation(self):
        # TODO: Abstract to an arbitrary number of cities
        # TODO: This should work recursively (somehow)
        if self.n == 3:
            self.circuit.h(0)
            self.circuit.cx(0, 1)
            self.circuit.x(0)
        elif self.n == 4:
            self.circuit.ry(np.arccos(-1/3), 0)
            self.circuit.ch(0, 1)
            self.circuit.ch(1, 2)
            self.circuit.cx(2, 3)
            self.circuit.cx(1, 2)
            self.circuit.cx(0, 1)
            self.circuit.x(0)
            self.circuit.ch(1, 2)
            self.circuit.cx(2, 5)
            self.circuit.cx(1, 2)
            self.circuit.ch(0, 3)
            self.circuit.cx(3, 4)
            self.circuit.cx(0, 3)
        else:
            pass

    def reverse_state_preparation(self):
        # TODO: Abstract to an arbitrary number of cities
        if self.n == 3:
            self.circuit.x(0)
            self.circuit.cx(0, 1)
            self.circuit.h(0)
        elif self.n == 4:
            self.circuit.cx(0, 3)
            self.circuit.cx(3, 4)
            self.circuit.ch(0, 3)
            self.circuit.cx(1, 2)
            self.circuit.cx(2, 5)
            self.circuit.ch(1, 2)
            self.circuit.x(0)
            self.circuit.cx(0, 1)
            self.circuit.cx(1, 2)
            self.circuit.cx(2, 3)
            self.circuit.ch(1, 2)
            self.circuit.ch(0, 1)
            self.circuit.ry(-np.arccos(-1/3), 0)
        else:
            pass

    def get_number_of_qubits(self) -> int:
        return (self.n-1)*(self.n-2)
    
    def build_phase_separator(self, gamma : qk.circuit.Parameter):
        for qubit, edge in enumerate(permutations(range(1, self.n), 2)):
            self.circuit.p(2 * gamma * (self.matrix[edge[0]][edge[1]] - (self.n-3)*(self.matrix[edge[0]][0]/(self.n-2)) - (self.n-3)*(self.matrix[0][edge[1]]/(self.n-2))), qubit)
            self.circuit.x(qubit)
            self.circuit.p(2 * gamma * (self.matrix[edge[0]][0]/(self.n-2) + self.matrix[0][edge[1]]/(self.n-2)), qubit)
            self.circuit.x(qubit)

    def build_mixer(self, beta : qk.circuit.Parameter):
        self.reverse_state_preparation()

        self.circuit.x(self.qubits)

        cp_gate = qk.circuit.library.standard_gates.PhaseGate(beta).control(len(self.qubits) - 1)
        self.circuit.append(cp_gate, self.qubits)

        self.circuit.x(self.qubits)

        self.build_state_preparation()

    def compute_path_length(self, string : str) -> float:
        value = 0
        for bit, edge in enumerate(permutations(range(1, self.n), 2)):
            bit_value = string[bit]
            if bit_value == "0":
                value += self.adj_matrix[edge[0]][0]/(self.n-2)
                value += self.adj_matrix[0][edge[1]]/(self.n-2)
            else:
                value -= (self.n-3)*(self.adj_matrix[edge[0]][0]/(self.n-2))
                value -= (self.n-3)*(self.adj_matrix[0][edge[1]]/(self.n-2))
                value += self.adj_matrix[edge[0]][edge[1]]
        return value

    def compute_expectation(self, counts : dict, shots : int) -> float:
        sum_count = 0
        for string, count in counts.items():
            sum_count += self.compute_path_length(string)*count

        return sum_count/shots
    
    def path_from_string(self, string : str) -> tuple:
        # n = int(0.5*(3 + (4*len(string) + 1)**0.5))
        n = self.n
        edge_in = set(range(1, n))
        edges = set()
        for bit, edge in enumerate(permutations(range(1, n), 2)):
            value = string[bit]
            if value == "1":
                edges.add(edge)
                edge_in.remove(edge[1])
        path = [0, edge_in.pop()]
        while edges:
            for edge in edges:
                if edge[0] == path[-1]:
                    path.append(edge[1])
                    edges.remove(edge)
                    break
        return path + [0]

    
