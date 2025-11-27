from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, List

import numpy as np
import pennylane as qml

from qvrp.core.qubo import QUBOBase
from qvrp.solver.types import SolverResult, Bitstring


@dataclass
class CommonConfig:
    """
    Common configuration shared by all PennyLane-based solvers.

    Attributes
    ----------
    device_name:
        PennyLane device name, e.g. 'default.qubit', 'lightning.qubit'.
    shots:
        Number of shots for sampling. If None, analytic mode is used
        for expectation-value circuits; sampling circuits should override this.
    interface:
        Differentiation interface ('autograd', 'jax', 'torch', etc.).
    diff_method:
        Differentiation method ('parameter-shift', 'adjoint', etc.).
    seed:
        Optional random seed for reproducibility.
    """
    device_name: str = "default.qubit"
    shots: Optional[int] = None
    interface: str = "autograd"
    diff_method: str = "parameter-shift"
    seed: Optional[int] = None


class SolverBase(ABC):
    """
    Base class for all PennyLane-based solvers (QAOA, VQE, etc.).

    It is backend-specific (PennyLane) but algorithm-agnostic.

    Responsibilities
    ----------------
    - Store the QUBO model (`QUBOBase`).
    - Configure a PennyLane device.
    - Build a Pauli-Z Hamiltonian from the Ising form of the QUBO.
    - Provide classical energy evaluation using QUBOBase.energy().
    """

    def __init__(
        self,
        qubo_model: QUBOBase,
        config: Optional[CommonConfig] = None,
    ) -> None:
        self.qubo_model = qubo_model
        self.num_qubits = qubo_model.num_vars

        self.config = config or CommonConfig()
        self._init_random_seed()

        self.device = qml.device(
            self.config.device_name,
            wires=self.num_qubits,
            shots=self.config.shots,
        )

        # Cost Hamiltonian for qubit-based algorithms
        self.cost_h = self._build_cost_hamiltonian_from_qubo()

    # ---------- Randomness / reproducibility ----------

    def _init_random_seed(self) -> None:
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    # ---------- QUBO → Pauli-Z Hamiltonian ----------

    def _build_cost_hamiltonian_from_qubo(self) -> qml.Hamiltonian:
        """
        Use the Ising representation (h, J, const) from QUBOBase.to_ising()
        to build a Pauli-Z Hamiltonian:

            H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

        The constant term is irrelevant for QAOA/VQE optimization and
        is therefore discarded.
        """
        h, J, const = self.qubo_model.to_ising()

        coeffs: list[float] = []
        ops: list[qml.operation.Operator] = []

        # Local fields h_i Z_i
        for i in range(self.num_qubits):
            if abs(h[i]) < 1e-12:
                continue
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

        # Couplings J_ij Z_i Z_j (i < j)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if abs(J[i, j]) < 1e-12:
                    continue
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

        if not coeffs:
            return qml.Hamiltonian([], [])

        return qml.Hamiltonian(coeffs, ops)

    # ---------- Classical utility ----------

    def classical_energy(self, bitstring: Bitstring) -> float:
        """
        Compute the QUBO energy E(x) using QUBOBase.energy().
        """
        return self.qubo_model.energy(bitstring)

    # ---------- Main interface for concrete solvers ----------

    @abstractmethod
    def run(self) -> SolverResult:
        """
        Execute the quantum optimization algorithm and return a SolverResult.
        Concrete solvers (QAOA, VQE, CV, etc.) must implement this method.
        """
        ...
