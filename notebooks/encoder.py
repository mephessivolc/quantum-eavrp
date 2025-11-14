from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable, Optional, Any, Iterable

import numpy as np
import networkx as nx

# Note: import pennylane only inside methods that actually need it,
# to keep dependencies flexible.


# ---------------------------------------------------------------------------
# QUBO data model
# ---------------------------------------------------------------------------

VarKey = Tuple[int, Hashable, Hashable]  # example: (vehicle_id, from_node, to_node)


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
        The actual key type is chosen by the Encoder.
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
        Export QUBO as a sparse dictionary mapping (i, j) -> coefficient with i <= j.

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
        ops: List[qml.operation.Operator] = []

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


# ---------------------------------------------------------------------------
# Encoder (VRP → QUBO, and bitstring → routes)
# ---------------------------------------------------------------------------


class Encoder:
    """
    VRP encoder: builds a QUBO model from a VRP instance and decodes bitstrings.

    Responsibilities
    ----------------
    - Store the VRP instance (graph, vehicles, depot, etc.).
    - Define binary variables (e.g., x_{kij} for vehicle k using arc i→j).
    - Build the QUBO with the objective + hard constraints.
    - Decode a bitstring back into vehicle routes.

    This class is domain-specific (VRP / EA-VRP).
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_vehicles: int,
        depot: Hashable,
        *,
        distance_attr: str = "distance",
        # additional VRP data can be passed via kwargs:
        demands: Optional[Dict[Hashable, float]] = None,
        capacity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.distance_attr = distance_attr

        self.demands = demands or {}
        self.capacity = capacity
        self.metadata = metadata or {}

        self.nodes: List[Hashable] = list(self.graph.nodes())
        if depot not in self.nodes:
            raise ValueError("Depot node is not in the graph.")

        self.customers: List[Hashable] = [n for n in self.nodes if n != depot]

        # Internal variable indexing (to be filled in _build_variable_index)
        self.var_index: Dict[VarKey, int] = {}
        self.index_var: Dict[int, VarKey] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_qubo(self) -> QUBO:
        """
        Build the QUBO model for this VRP instance.

        This includes:
            - objective term (e.g., distance minimization),
            - hard constraints (e.g., degree constraints, depot constraints),
        encoded as penalties in the Q matrix.

        Returns
        -------
        QUBO
            The QUBO model containing Q, constant, and variable mappings.
        """
        self._build_variable_index()
        n_vars = len(self.var_index)
        Q = np.zeros((n_vars, n_vars), dtype=float)
        constant = 0.0

        # 1. Objective: distance minimization
        constant = self._encode_objective(Q, constant)

        # 2. Hard constraints: must be satisfied by any valid solution
        constant = self._encode_hard_constraints(Q, constant)

        return QUBO(Q=Q, constant=constant, var_index=self.var_index, index_var=self.index_var)

    def decode(self, bitstring: Iterable[int]) -> Dict[int, Any]:
        """
        Decode a bitstring into a set of routes (one per vehicle).

        Parameters
        ----------
        bitstring : iterable of int
            Binary values corresponding to QUBO variables.

        Returns
        -------
        Dict[int, Any]
            A structured representation of routes, e.g.:
            {
                0: {"route": [depot, ..., depot], "distance": ...},
                1: {"route": [depot, ..., depot], "distance": ...},
                ...
            }
        """
        # TODO: implement bitstring → arcs → routes reconstruction.
        #       This will likely:
        #       - build a directed graph per vehicle from active x_{kij} = 1,
        #       - extract cycles/paths starting and ending at the depot,
        #       - compute distances using the original graph.
        raise NotImplementedError("Decoder not implemented yet.")

    @property
    def num_variables(self) -> int:
        """Return the number of binary variables (after indexing has been built)."""
        if not self.var_index:
            # Optionally, we could call _build_variable_index() here.
            raise RuntimeError("Variable index has not been built yet.")
        return len(self.var_index)

    # ------------------------------------------------------------------
    # Internal helpers (encoding)
    # ------------------------------------------------------------------

    def _build_variable_index(self) -> None:
        """
        Assign a unique index to each binary variable used by the QUBO.

        Typical choice for arc-based VRP:
            VarKey = (k, i, j) for vehicle k going from node i to node j.
        """
        if self.var_index:
            # Already built; nothing to do
            return

        idx = 0
        for k in range(self.num_vehicles):
            for i in self.nodes:
                for j in self.nodes:
                    if i == j:
                        continue
                    key: VarKey = (k, i, j)
                    self.var_index[key] = idx
                    self.index_var[idx] = key
                    idx += 1

    def _encode_objective(self, Q: np.ndarray, constant: float) -> float:
        """
        Encode the VRP objective (e.g., total travel distance) into Q.

        Parameters
        ----------
        Q : np.ndarray
            QUBO matrix to be updated in-place.
        constant : float
            Current constant term, to be updated and returned.

        Returns
        -------
        float
            Updated constant term.
        """
        # TODO: add distance-based contributions to Q diagonal entries.
        #       For each vehicle k and edge (i, j), add dist(i, j) * x_{kij}.
        #       That contributes to Q[idx, idx].
        raise NotImplementedError("Objective encoding not implemented yet.")

    def _encode_hard_constraints(self, Q: np.ndarray, constant: float) -> float:
        """
        Encode all hard constraints as quadratic penalties added to Q.

        Examples:
            - Each customer has exactly one incoming and one outgoing edge.
            - Each vehicle leaves the depot once and returns once.

        Parameters
        ----------
        Q : np.ndarray
            QUBO matrix to be updated in-place.
        constant : float
            Current constant term, to be updated and returned.

        Returns
        -------
        float
            Updated constant term including contributions from constraints.
        """
        # TODO: implement constraint penalties using helper methods like:
        #       - _add_equality_one_constraint(indices, penalty, Q, constant)
        #       This is where you add "hard" VRP structure.
        raise NotImplementedError("Hard constraints encoding not implemented yet.")

    # You can add low-level helpers like `_add_qubo_term` and
    # `_add_equality_one_constraint` here.