"""
QUBO data model and utilities.

This module is domain-agnostic: it does not know about VRP or graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

from pennylane import numpy as np


# Semantic type for variable keys can be customized by the Encoder.
VarKey = Tuple[int, Any, Any]  # e.g., (vehicle_id, from_node, to_node)


@dataclass
class QUBO:
    """
    Generic QUBO model container.

    Represents an energy function:
        E(x) = x^T Q x + constant
    with x ∈ {0, 1}^n.

    Attributes
    ----------
    Q : np.ndarray
        Symmetric QUBO matrix of shape (n_vars, n_vars).
    constant : float
        Constant offset term in the objective (does not affect argmin).
    var_index : Dict[VarKey, int]
        Mapping from semantic variable keys to indices in Q.
    index_var : Dict[int, VarKey]
        Reverse mapping from index to semantic variable keys.
    """

    Q: np.ndarray
    constant: float
    var_index: Dict[VarKey, int]
    index_var: Dict[int, VarKey]

    @property
    def num_variables(self) -> int:
        """Return the number of binary variables (qubits)."""
        return self.Q.shape[0]

    def to_qubo_dict(self, tol: float = 0.0) -> Dict[Tuple[int, int], float]:
        """
        Export QUBO as a sparse dictionary mapping (i, j) -> coefficient.

        Parameters
        ----------
        tol : float
            Threshold below which coefficients are ignored.

        Returns
        -------
        Dict[(int, int), float]
            Dictionary representation suitable for classical QUBO solvers.
        """
        n = self.num_variables
        qubo: Dict[Tuple[int, int], float] = {}

        for i in range(n):
            for j in range(i, n):
                val = float(self.Q[i, j])
                if abs(val) > tol:
                    qubo[(i, j)] = val
        return qubo

    def to_hamiltonian(self, tol: float = 0.0):
        """
        Convert this QUBO into a PennyLane Ising Hamiltonian H(Z).

        The mapping used is:
            x_i = (1 - Z_i) / 2

        The constant shift is ignored (does not affect the optimal bitstring).

        Parameters
        ----------
        tol : float
            Threshold below which coefficients are ignored.

        Returns
        -------
        qml.Hamiltonian
            PennyLane Hamiltonian built from PauliZ and PauliZ ⊗ PauliZ terms.
        """
        import pennylane as qml

        Q = self.Q
        n = self.num_variables

        # Linear and quadratic Ising coefficients
        h = np.zeros(n, dtype=float)
        J: Dict[Tuple[int, int], float] = {}

        # Compute linear terms
        for i in range(n):
            qii = float(Q[i, i])
            off_sum = float(Q[i, :].sum() - qii)
            h[i] = -qii / 2.0 - off_sum / 4.0

        # Compute pairwise couplings
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(Q[i, j])
                if abs(coeff) > tol:
                    J[(i, j)] = coeff / 4.0

        coeffs: List[float] = []
        ops: List[Any] = []

        # Add Z_i terms
        for i in range(n):
            if abs(h[i]) > tol:
                coeffs.append(float(h[i]))
                ops.append(qml.PauliZ(wires=i))

        # Add Z_i Z_j terms
        for (i, j), jij in J.items():
            if abs(jij) > tol:
                coeffs.append(float(jij))
                ops.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))

        if not coeffs:
            coeffs = [0.0]
            ops = [qml.Identity(wires=0)]

        return qml.Hamiltonian(coeffs, ops)

    def human_readable(
        self,
        tol: float = 0.0,
        max_terms: Optional[int] = None,
    ) -> str:
        """
        Build a human-readable representation of the QUBO:

            E(x) = constant + sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j

        Parameters
        ----------
        tol : float
            Threshold below which coefficients are ignored.
        max_terms : int, optional
            If given, truncate after at most max_terms non-zero terms.

        Returns
        -------
        str
            A symbolic equation-like string.
        """
        n = self.num_variables

        def var_name(i: int) -> str:
            key = self.index_var.get(i, None)
            if isinstance(key, tuple) and len(key) == 3:
                k, u, v = key
                return f"x_{k}_{u}_{v}"
            return f"x_{i}"

        terms: List[str] = []
        term_count = 0

        # Constant term
        if abs(self.constant) > tol:
            terms.append(f"{self.constant:.4f}")
            term_count += 1

        # Linear terms
        for i in range(n):
            coeff = float(self.Q[i, i])
            if abs(coeff) <= tol:
                continue
            if max_terms is not None and term_count >= max_terms:
                break
            name = var_name(i)
            sign = "+" if coeff >= 0 else "-"
            mag = abs(coeff)
            if not terms:
                prefix = "-" if coeff < 0 else ""
                terms.append(f"{prefix}{mag:.4f}*{name}")
            else:
                terms.append(f"{sign} {mag:.4f}*{name}")
            term_count += 1

        # Quadratic terms
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(self.Q[i, j])
                if abs(coeff) <= tol:
                    continue
                if max_terms is not None and term_count >= max_terms:
                    break
                vi = var_name(i)
                vj = var_name(j)
                sign = "+" if coeff >= 0 else "-"
                mag = abs(coeff)
                if not terms:
                    prefix = "-" if coeff < 0 else ""
                    terms.append(f"{prefix}{mag:.4f}*{vi}*{vj}")
                else:
                    terms.append(f"{sign} {mag:.4f}*{vi}*{vj}")
                term_count += 1
            if max_terms is not None and term_count >= max_terms:
                break

        if not terms:
            return "E(x) = 0"
        return "E(x) = " + " ".join(terms)
