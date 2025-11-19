"""
Solver interfaces and QAOA-based implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from pennylane import numpy as np

from qubo import QUBO


@dataclass
class SolverResult:
    """
    Container for the result returned by a Solver.

    Attributes
    ----------
    bitstring : np.ndarray
        Binary vector representing the chosen solution.
    energy : float
        Objective value (Hamiltonian expectation or best sample energy).
    extra : Dict[str, Any]
        Optional extra information (parameters, history, raw samples, etc.).
    """

    bitstring: np.ndarray
    energy: float
    extra: Dict[str, Any]


class BaseSolver:
    """
    Abstract base class for QUBO/Ising solvers.

    Concrete solvers (e.g., QAOASolver) should implement the `solve` method.
    """

    def solve(self, qubo: QUBO) -> SolverResult:
        """
        Solve the given QUBO instance and return the best solution found.

        Parameters
        ----------
        qubo : QUBO
            The QUBO model to be solved.

        Returns
        -------
        SolverResult
            The solution bitstring and associated energy.
        """
        raise NotImplementedError


class QAOASolver(BaseSolver):
    """
    QAOA-based solver using PennyLane.

    This class is responsible for:
        - building the QAOA circuit from the given cost Hamiltonian,
        - defining and applying the mixer Hamiltonian,
        - optimizing the variational parameters,
        - extracting the best bitstring from samples or expectation.
    """

    def __init__(
        self,
        p: int = 1,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        optimizer: Optional[Any] = None,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.p = p
        self.device_name = device
        self.shots = shots
        self.optimizer = optimizer  # e.g., qml.GradientDescentOptimizer()
        self.max_steps = max_steps
        self.seed = seed

    def _build_mixer_hamiltonian(self, num_qubits: int):
        """
        Build the mixer Hamiltonian H_M.

        Default choice: X-mixer, i.e.,
            H_M = sum_i X_i
        """
        import pennylane as qml

        coeffs = [1.0] * num_qubits
        ops = [qml.PauliX(wires=i) for i in range(num_qubits)]
        return qml.Hamiltonian(coeffs, ops)

    def solve(self, qubo: QUBO) -> SolverResult:
        """
        Solve the given QUBO using QAOA.

        Parameters
        ----------
        qubo : QUBO
            The QUBO model to be solved.

        Returns
        -------
        SolverResult
            Best bitstring and corresponding energy.
        """
        import pennylane as qml

        H_cost = qubo.to_hamiltonian()
        n = qubo.num_variables

        H_mixer = self._build_mixer_hamiltonian(n)

        dev = qml.device(self.device_name, wires=n, shots=self.shots, seed=self.seed)

        @qml.qnode(dev)
        def circuit(gammas, betas):
            # Initial state |+>^{⊗n}
            for i in range(n):
                qml.Hadamard(wires=i)

            # QAOA layers
            for layer in range(self.p):
                # Cost unitary: exp(-i gamma_l H_cost)
                qml.ApproxTimeEvolution(H_cost, gammas[layer], 1)

                # Mixer unitary: exp(-i beta_l H_mixer)
                qml.ApproxTimeEvolution(H_mixer, betas[layer], 1)

            return qml.expval(H_cost)

        # Initialize parameters
        gammas = np.random.uniform(0, 2 * np.pi, size=(self.p,))
        betas = np.random.uniform(0, 2 * np.pi, size=(self.p,))

        opt = self.optimizer or qml.GradientDescentOptimizer(stepsize=0.1)

        def cost_fn(params):
            g, b = params[: self.p], params[self.p :]
            return circuit(g, b)

        params = np.concatenate([gammas, betas])

        history = []
        for _ in range(self.max_steps):
            params, energy = opt.step_and_cost(cost_fn, params)
            history.append(float(energy))

        # Final parameters
        gammas_opt = params[: self.p]
        betas_opt = params[self.p :]

        # Build a sampling circuit to extract a bitstring
        @qml.qnode(dev)
        def sampling_circuit(gammas, betas):
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(self.p):
                qml.ApproxTimeEvolution(H_cost, gammas[layer], 1)
                qml.ApproxTimeEvolution(H_mixer, betas[layer], 1)
            return qml.sample(qml.PauliZ(wires=list(range(n))))

        # Sample in Z basis and convert ±1 to {0, 1}
        samples = sampling_circuit(gammas_opt, betas_opt)
        # samples has shape (shots, n) if shots is not None, otherwise (n,)
        if self.shots is None:
            # Single sample (deterministic expectation); convert directly
            z_vals = np.array(samples)
            bitstring = ((1 - z_vals) / 2).astype(int)
        else:
            # Take the first sample as representative (or use majority vote)
            z_vals = np.array(samples[0])
            bitstring = ((1 - z_vals) / 2).astype(int)

        final_energy = float(circuit(gammas_opt, betas_opt))

        return SolverResult(
            bitstring=bitstring,
            energy=final_energy,
            extra={
                "history": history,
                "gammas": gammas_opt,
                "betas": betas_opt,
            },
        )
