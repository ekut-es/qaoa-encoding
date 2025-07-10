import qiskit as qk
import numpy as np

from QAOA_TSP_Alternating import QAOA_TSP_Alternating

class QAOA_TSP_Grover(QAOA_TSP_Alternating):

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

    def build_mixer(self, beta : qk.circuit.Parameter):
        self.reverse_state_preparation()

        self.circuit.x(self.qubits)

        cp_gate = qk.circuit.library.standard_gates.PhaseGate(beta).control(len(self.qubits) - 1)
        self.circuit.append(cp_gate, self.qubits)

        self.circuit.x(self.qubits)

        self.build_state_preparation()