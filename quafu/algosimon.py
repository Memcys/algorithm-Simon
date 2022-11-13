# Implement the Simon's algorithm, and run it on Quafu quantum cloud computing platform http://quafu.baqis.ac.cn/.
from dataclasses import dataclass, KW_ONLY
import random
import logging

import numpy as np
from quafu import QuantumCircuit, Task, simulate

from sage.matrix.constructor import matrix
from sage.rings.finite_rings.finite_field_constructor import GF


class IntStrArray:
    @staticmethod
    def int2str(i: int, l: int=0) -> str:
        s = bin(i)[2:]

        if l > (ls := len(s)):
            s = '0' * (l - ls) + s

        return s

    @staticmethod
    def str2qstr(s: str, l: int=0) -> str:
        rs = s[::-1]

        if l > (ls := len(rs)):
            rs += '0' * (l - ls)

        return rs

    @classmethod
    def int2qstr(cls, i: int, l: int) -> str:
        s = cls.int2str(i)
        qs = cls.str2qstr(s)
        qs += '0' * (l -len(qs))

        return qs

    @classmethod
    def int2array(cls, i: int, l: int) -> np.ndarray:
        s = cls.int2str(i, l)
        a = cls.str2array(s)

        return a

    @classmethod
    def int2qarray(cls, i: int, l: int) -> np.ndarray:
        qs = cls.int2qstr(i, l)
        qa = Simon.str2array(qs)

        return qa

    @staticmethod
    def str2array(s: str) -> np.ndarray:
        """Convert a str to an ndarray.
        
        E.g., s='10100' -> array([1, 0, 1, 0, 0]).
        """
        return np.array([int(c) for c in s])


@dataclass
class Simon(IntStrArray):
    """Quantum circuit for the Simon algorithm.

    This class is specialized for the Quafu quantum cloud computing platform (http://quafu.baqis.ac.cn/).
    """
    _: KW_ONLY

    # * the following will be keyword only
    secretb: int = -1   # ! integer secretb should be non-negative
    Ndata: int = 5      # * 2*Ndata should be no larger than the number of qubits available
    # bstr: str = ''

    @property
    def Nqubit(self) -> int:
        """Number of qubits in use,
        including both data qubits and ancilla qubits (of the same size).
        """
        return 2 * self.Ndata

    @property
    def bstr(self) -> str:
        """secretb in binary string (in normal order).
        E.g., secretb=2 ==> (implies that) bstr='10'.
        """
        return self.int2str(i=self.secretb)

    @property
    def qstr(self) -> str:
        """secretb in qubit string (in reverse order).
        The length will be Ndata.

        E.g., secret=2, Ndata=5 -> qstr='01000'.
        """
        b = self.secretb
        Ndata = self.Ndata

        q = self.int2qstr(i=b, l=Ndata)

        return q

    @property
    def qarray(self) -> np.ndarray:
        """secretb in qubit array (in reverse order).
        The length will be Ndata.

        E.g., secret=2, Ndata=5 -> qstr=array([0, 1, 0, 0, 0]).
        """
        b = self.secretb
        Ndata = self.Ndata

        a = self.int2qarray(i=b, l=Ndata)

        return a


    @property
    def qcircuit(self):
        """Quantu circuit of Simon's algorithm."""
        return self.quantum_circuit()

    
    def __post_init__(self):
        if self.secretb < 0:
            self.secretb = self._random_b()
            # self.bstr = self._random_bstr()
            # self.secretb = self._bstr2int()

        # self.qcircuit = self.quantum_circuit()

    def _random_b(self) -> int:
        """Generate a random integer b.

        Note that 0 <= b <= 2^Ndata - 1.
        """
        Ndata = self.Ndata
        b = random.randint(0, 2**Ndata - 1)

        return b

    # def _qstr2array(self):
    #     vec = self.str2array(self.qstr)

    #     return vec

    def quantum_circuit(self):
        """Prepare the quantum circuit."""
        Ndata = self.Ndata
        Nqubit = self.Nqubit
        qstr = self.qstr

        q = QuantumCircuit(Nqubit)
        
        # Hadamard gates on the first half qubits
        for i in range(Ndata):
            q.h(i)

        for i in range(Ndata):
            q.cnot(i, i+Ndata)


        try:
            # in this case, secretb != 0

            # last nonzero bit of bstr, i.e.,
            # first nonzero bit of qstr
            bstrLastnonzero = qstr.index('1')

            for i in range(bstrLastnonzero, Ndata):
                if qstr[i] == '1':
                    q.cnot(bstrLastnonzero, i+Ndata)
        finally:
            for i in range(Ndata):
                q.h(i)

            q.measure(range(Ndata))

        return q

    def run(self, shots: int=4000, compile: bool=True):
        """
        Run the quantum circuit on Quafu quantum cloud computing platform(http://quafu.baqis.ac.cn/).
        
        :param self: Access the class attributes and methods
        :param shots:int=4000: Set the number of shots to be taken
        :param compile:bool=True: Compile the circuit to a more efficient representation
        :return: A task object
        :doc-author: Trelent
        """
        """Run the quantum circuit on Quafu.
        

        int shots: """
        task = Task()
        task.load_account()
        task.config(backend="ScQ-P10", shots=shots, compile=compile)

        logging.info("Your quantum circuit will be sent to ScQ-P10 on Quafu.\n\nPlease wait for a minute.")
        
        result = task.send(self.qcircuit)

        logging.info("Done. Please check your results.")
        
        return result

    def draw_circuit(self, width: int=2):
        """Draw the quantum circuit."""
        self.qcircuit.draw_circuit(width=width)

    @staticmethod
    def sortcounts(d: dict) -> list:
        """Sort a dict in descending order by its values.

        Return the sorted keys (strings) and values (counts), respectively.
        """
        sorted_dict = sorted(
            d.items(),
            key=lambda kv: (kv[1], kv[0]),
            reverse=True
        )

        keyval = lambda d, j: [
            d[i][j] for i in range(len(d))
        ]
        key = lambda d: keyval(d, 0)
        value = lambda d: keyval(d, 1)

        strings = key(sorted_dict)
        strings_arr = np.array(list(map(Simon.str2array, strings)))

        counts_arr = np.array(value(sorted_dict))

        return (strings_arr, counts_arr)

    @staticmethod
    def dotmod2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return an array of boolean results of a·b == 0 mod 2.
        """
        return np.dot(a, b) % 2


@dataclass
class CheckCircuit:
    def __init__(self, runresult) -> None:
        qubitcounts = runresult.counts
        self.mat, self.counts = Simon.sortcounts(qubitcounts)
        
    @staticmethod
    def simulate(simon: Simon):
        qc = simon.qcircuit
        Ndata = simon.Ndata

        simu = simulate(qc, output="amplitudes")
        intq = simu.amplitudes.nonzero()[0]
        
        X = np.array([
            simon.int2array(i=i, l=Ndata) \
            for i in intq
        ])

        return (X, simu)

    @staticmethod
    def solve(borthogonal: np.ndarray) -> np.ndarray:
        """Solve X · b^T = 0 for b."""
        nrow, ncol = borthogonal.shape
        X = matrix(
            ring=GF(2),
            nrows=nrow,
            ncols=ncol,
            entries=borthogonal
        )

        kernel = X.right_kernel()

        b = kernel.basis_matrix()
        # barr = b.matrix().numpy()
        # bstr = .array2string()

        return b


    def Nmax(self, simon: Simon):
        """The max number that the Nmax most significant run results are orthogonal to qstr.
        
        Suppose you know the secret b, N should always be no larger than Nmax.
        """
        boolres = simon.dotmod2(self.mat, simon.qarray)
        Nmax = boolres.nonzero()[0][0]

        return Nmax


    @classmethod
    def check(cls, simon: Simon):
        """Check if the quantum circuit functions as expected.

        Return True if it does, False otherwise.
        """
        X, _ = cls.simulate(simon=simon)
        b = simon.qarray

        nbool = np.sum(
            simon.dotmod2(X, b)
        )

        return nbool == 0