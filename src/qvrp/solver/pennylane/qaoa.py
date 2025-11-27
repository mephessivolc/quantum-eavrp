from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pennylane as qml

from qvrp.core.qubo import QUBOBase
from qvrp.solver.types import SolverResult, Counts
from qvrp.solver.pennylane.common import SolverBase, CommonConfig


@dataclass
class QAOAConfig:
    layers: int = 2
    max_iterations: int = 100
    stepsize: float = 0.1
    shots: int = 10_000
    seed: int | None = None
    device_name: str = "lightning.qubit"


class QAOASolver(SolverBase):

    def __init__(self, qubo_model: QUBOBase, config: QAOAConfig) -> None:
        common_cfg = CommonConfig(
            device_name=config.device_name,
            shots=None,          # modo analítico para otimização
            seed=config.seed,
        )
        super().__init__(qubo_model=qubo_model, config=common_cfg)

        self.cfg = config
        self.num_layers = config.layers
        self.num_params = 2 * self.num_layers  # [gammas..., betas...]
        self.expectation_qnode = self._build_expectation_qnode()

    # --- QAOA: circuito para <H_cost> ---

    def _build_expectation_qnode(self):
        dev = self.device
        cost_h = self.cost_h
        num_qubits = self.num_qubits
        p = self.num_layers

        @qml.qnode(dev, interface=self.config.interface, diff_method=self.config.diff_method)
        def circuit(params):
            gammas = params[:p]
            betas = params[p:]

            for w in range(num_qubits):
                qml.Hadamard(wires=w)

            for layer in range(p):
                g = gammas[layer]
                b = betas[layer]
                qml.templates.ApproxTimeEvolution(cost_h, g, n=1)
                for w in range(num_qubits):
                    qml.RX(2.0 * b, wires=w)

            return qml.expval(cost_h)

        return circuit

    # --- Otimização ---

    def _init_params(self) -> Any:
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
        return 0.01 * np.random.randn(self.num_params)

    def _optimize(self) -> Dict[str, Any]:
        params = self._init_params()
        optimizer = qml.GradientDescentOptimizer(stepsize=self.cfg.stepsize)

        history: Dict[str, list] = {"params": [], "energies": []}

        for _ in range(self.cfg.max_iterations):
            params, energy = optimizer.step_and_cost(self.expectation_qnode, params)
            history["params"].append(params.copy())
            history["energies"].append(float(energy))

        return {"optimal_params": params, "history": history}

    # --- Amostragem ---

    def _build_sampling_qnode(self):
        sample_dev = qml.device(
            self.config.device_name,
            wires=self.num_qubits,
            shots=self.cfg.shots,
        )
        cost_h = self.cost_h
        num_qubits = self.num_qubits
        p = self.num_layers

        @qml.qnode(sample_dev, interface=self.config.interface)
        def circuit(params):
            gammas = params[:p]
            betas = params[p:]

            for w in range(num_qubits):
                qml.Hadamard(wires=w)

            for layer in range(p):
                g = gammas[layer]
                b = betas[layer]
                qml.templates.ApproxTimeEvolution(cost_h, g, n=1)
                for w in range(num_qubits):
                    qml.RX(2.0 * b, wires=w)

            return [qml.sample(qml.PauliZ(w)) for w in range(num_qubits)]

        return circuit

    def _sample_bitstrings(self, optimal_params: Any) -> Counts:
        sampling_circuit = self._build_sampling_qnode()
        samples = sampling_circuit(optimal_params)  # (num_qubits, shots)
        samples = np.array(samples).T

        counts: Counts = {}
        for row in samples:
            bits = ((1 - row) // 2).astype(int)  # +1->0, -1->1
            s = "".join(str(b) for b in bits)
            counts[s] = counts.get(s, 0) + 1

        return counts

    # --- API pública ---

    def run(self) -> SolverResult:
        opt_data = self._optimize()
        optimal_params = opt_data["optimal_params"]
        counts = self._sample_bitstrings(optimal_params)

        best_str = max(counts, key=counts.get)
        bitstring = [int(b) for b in best_str]
        energy = self.classical_energy(bitstring)

        metadata = {
            "counts": counts,
            "optimizer_history": opt_data["history"],
            "optimal_params": optimal_params,
        }

        return SolverResult(
            bitstring=bitstring,
            energy=energy,
            raw_samples=counts,
            metadata=metadata,
        )