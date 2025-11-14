from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Hashable, Optional

import numpy as np
import networkx as nx


VarKey = Tuple[int, Hashable, Hashable]  # (vehicle_id, from_node, to_node)


@dataclass
class QUBOModel:
    """
    Container for a QUBO model: E(x) = x^T Q x + constant.

    Attributes
    ----------
    Q : np.ndarray
        Symmetric QUBO matrix of shape (n_vars, n_vars).
    constant : float
        Constant offset term in the objective.
    var_index : Dict[VarKey, int]
        Mapping from variable key (e.g., (vehicle, i, j)) to index in Q.
    index_var : Dict[int, VarKey]
        Reverse mapping from index to variable key.
    """
    Q: np.ndarray
    constant: float
    var_index: Dict[VarKey, int]
    index_var: Dict[int, VarKey]

    @property
    def num_variables(self) -> int:
        """
            It is the number of qubits needed
        """
        return self.Q.shape[0]

    def to_qubo_dict(self, tol: float = 0.0) -> Dict[Tuple[int, int], float]:
        """
        Export QUBO as a dictionary mapping (i, j) -> coefficient (i <= j).

        Parameters
        ----------
        tol : float
            Coefficients with absolute value <= tol are dropped.

        Returns
        -------
        Dict[(int, int), float]
        """
        n = self.num_variables
        qubo: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i, n):
                val = self.Q[i, j]
                if abs(val) > tol:
                    qubo[(i, j)] = float(val)
        return qubo

    def to_hamiltonian(self, tol: float = 0.0):
        """
        Convert the QUBO into a PennyLane Ising Hamiltonian H(Z).

        The QUBO is E(x) = x^T Q x + constant with x ∈ {0, 1}^n.
        Using x_i = (1 - Z_i)/2, we obtain:

            H = const' * I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

        where:
            h_i = -Q_ii/2 - (1/4) * sum_{j != i} Q_ij
            J_ij = Q_ij / 4 for i < j

        The constant shift is kept internally but is irrelevant for optimization.

        Parameters
        ----------
        tol : float
            Coefficients with absolute value <= tol are dropped.

        Returns
        -------
        qml.Hamiltonian
            PennyLane Hamiltonian built from PauliZ and PauliZ ⊗ PauliZ terms.
        """
        import pennylane as qml

        Q = self.Q
        n = self.num_variables

        # Compute Ising coefficients
        h = np.zeros(n, dtype=float)
        J: Dict[Tuple[int, int], float] = {}

        # Linear terms h_i
        for i in range(n):
            qii = Q[i, i]
            # Sum of off-diagonal Q_ij (symmetric)
            off_sum = float(Q[i, :].sum() - qii)
            h[i] = -qii / 2.0 - off_sum / 4.0

        # Quadratic terms J_ij
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > tol:
                    J[(i, j)] = Q[i, j] / 4.0

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

        # Constant shift from QUBO can be added if needed; usually not required
        # for QAOA because it does not affect the optimal bitstring.

        if not coeffs:
            # Zero Hamiltonian (edge case)
            coeffs = [0.0]
            ops = [qml.Identity(wires=0)]

        return qml.Hamiltonian(coeffs, ops)
    
    def _default_var_name(self, idx: int) -> str:
        """
        Default human-readable name for variable at position idx.

        For VRP with VarKey = (vehicle, i, j), this becomes:
            x_v_i_j
        """
        key = self.index_var[idx]  # e.g. (k, i, j)
        if isinstance(key, tuple) and len(key) == 3:
            k, i, j = key
            return f"x_{k}_{i}_{j}"
        # Fallback if the key has different structure
        return f"x_{idx}"

    def to_human_readable(
        self,
        tol: float = 0.0,
        var_namer: Optional[Callable[[int, VarKey], str]] = None,
        max_terms: Optional[int] = None,
    ) -> str:
        """
        Build a human-readable representation of the QUBO:

            E(x) = constant
                   + sum_i Q_ii * x_i
                   + sum_{i<j} Q_ij * x_i * x_j

        Parameters
        ----------
        tol : float
            Coefficients with |value| <= tol are ignored.
        var_namer : callable, optional
            Function (idx, key) -> variable name.
            If None, use the internal default naming.
        max_terms : int, optional
            If given, truncate after at most max_terms non-zero terms
            (useful for large problems).

        Returns
        -------
        str
            A human-readable equation string.
        """
        n = self.num_variables

        def name_var(i: int) -> str:
            key = self.index_var[i]
            if var_namer is not None:
                return var_namer(i, key)
            return self._default_var_name(i)

        terms = []
        term_count = 0

        # Constant term
        if abs(self.constant) > tol:
            terms.append(f"{self.constant:.4f}")
            term_count += 1

        # Linear terms: Q_ii * x_i
        for i in range(n):
            coeff = float(self.Q[i, i])
            if abs(coeff) <= tol:
                continue
            if max_terms is not None and term_count >= max_terms:
                break
            var_name = name_var(i)
            sign = "+" if coeff >= 0 else "-"
            coeff_abs = abs(coeff)
            if not terms:
                # First term: do not print leading '+'
                prefix = "-" if coeff < 0 else ""
                terms.append(f"{prefix}{coeff_abs:.4f}*{var_name}")
            else:
                terms.append(f"{sign} {coeff_abs:.4f}*{var_name}")
            term_count += 1

        # Quadratic terms: Q_ij * x_i * x_j, i < j
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(self.Q[i, j])
                if abs(coeff) <= tol:
                    continue
                if max_terms is not None and term_count >= max_terms:
                    break
                var_i = name_var(i)
                var_j = name_var(j)
                sign = "+" if coeff >= 0 else "-"
                coeff_abs = abs(coeff)
                if not terms:
                    prefix = "-" if coeff < 0 else ""
                    terms.append(f"{prefix}{coeff_abs:.4f}*{var_i}*{var_j}")
                else:
                    terms.append(f"{sign} {coeff_abs:.4f}*{var_i}*{var_j}")
                term_count += 1
            if max_terms is not None and term_count >= max_terms:
                break

        if not terms:
            return "E(x) = 0"

        return "E(x) = " + " ".join(terms)


class Encoder:
    """
    Build a QUBO model for a simple multi-vehicle VRP from a NetworkX graph.

    Model assumptions
    -----------------
    - One depot node (given explicitly).
    - |K| vehicles, each starting/ending at the depot.
    - Each customer must be visited exactly once in total (over all vehicles).
    - No capacity constraints yet.
    - No explicit sub-tour elimination yet (degree constraints only).
    - Variables x_{kij} = 1 if vehicle k travels directly from node i to node j.
      We exclude i == j.

    The graph is assumed to be complete (or at least contain all needed edges),
    with edge attribute `distance_attr` giving the travel cost.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_vehicles: int,
        depot: Hashable,
        distance_attr: str = "distance",
        objective_scale: float = 1.0,
        penalty_visit: float = 10.0,
        penalty_depot: float = 10.0,
    ) -> None:
        self.graph = graph
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.distance_attr = distance_attr

        self.objective_scale = float(objective_scale)
        self.penalty_visit = float(penalty_visit)
        self.penalty_depot = float(penalty_depot)

        # Derived sets
        self.nodes: List[Hashable] = list(self.graph.nodes())
        if depot not in self.nodes:
            raise ValueError("Depot node is not in the graph.")

        self.customers: List[Hashable] = [n for n in self.nodes if n != depot]

        # Variable index maps
        self.var_index: Dict[VarKey, int] = {}
        self.index_var: Dict[int, VarKey] = {}

        self._build_variable_index()
        n_vars = len(self.var_index)
        self.Q = np.zeros((n_vars, n_vars), dtype=float)
        self.constant: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_qubo(self) -> QUBOModel:
        """
        Build the full QUBO: objective + constraints.

        Returns
        -------
        QUBOModel
        """
        self._build_objective()
        self._build_visit_constraints()
        self._build_depot_constraints()
        return QUBOModel(
            Q=self.Q,
            constant=self.constant,
            var_index=self.var_index,
            index_var=self.index_var,
        )

    # ------------------------------------------------------------------
    # Variable indexing
    # ------------------------------------------------------------------

    def _build_variable_index(self) -> None:
        """
        Create a dense index for each binary variable x_{kij}.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers to update Q
    # ------------------------------------------------------------------

    def _add_qubo_term(self, i: int, j: int, value: float) -> None:
        """
        Add value to Q_{ij} (and Q_{ji} if i != j), enforcing symmetry.
        """
        if i == j:
            self.Q[i, i] += value
        else:
            self.Q[i, j] += value
            self.Q[j, i] += value

    def _add_constraint_equality_one(self, indices: List[int], penalty: float) -> None:
        """
        Add penalty * (1 - sum_{i in indices} x_i)^2 to the QUBO.

        For rhs = 1 and binary x_i, the expansion gives:
            penalty * [1 - 2 * sum x_i + (sum x_i)^2]
        with (sum x_i)^2 = sum x_i + 2 * sum_{i<j} x_i x_j.

        So:
            constant += penalty
            linear:   Q_ii += penalty * (-1)  for each i in indices
            quad:     Q_ij += penalty * 2     for i < j in indices
        """
        if not indices:
            return

        # Constant term
        self.constant += penalty

        # Linear terms
        for i in indices:
            self._add_qubo_term(i, i, -penalty)

        # Quadratic terms
        for idx_a in range(len(indices)):
            for idx_b in range(idx_a + 1, len(indices)):
                i = indices[idx_a]
                j = indices[idx_b]
                self._add_qubo_term(i, j, 2.0 * penalty)

    # ------------------------------------------------------------------
    # Objective: distance minimization
    # ------------------------------------------------------------------

    def _build_objective(self) -> None:
        """
        Build the objective term: minimize total travel distance.

        Adds terms:
            sum_{k} sum_{(i,j)} distance(i, j) * x_{kij}
        which correspond to diagonal QUBO entries.
        """
        for (i, j, data) in self.graph.edges(data=True):
            # Get distance/cost for undirected edge, but create both directions
            dist = data.get(self.distance_attr, data.get("weight", 1.0))
            dist = float(dist) * self.objective_scale

            for k in range(self.num_vehicles):
                # i -> j
                if i != j:
                    idx_ij = self.var_index[(k, i, j)]
                    self._add_qubo_term(idx_ij, idx_ij, dist)

                # j -> i (if graph is undirected, treat as separate variable)
                if j != i:
                    idx_ji = self.var_index[(k, j, i)]
                    self._add_qubo_term(idx_ji, idx_ji, dist)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _build_visit_constraints(self) -> None:
        """
        Each customer must have exactly one incoming and one outgoing edge
        over all vehicles.

        For every customer node c:
            sum_{k} sum_{i != c} x_{k i c} = 1
            sum_{k} sum_{j != c} x_{k c j} = 1
        """
        for c in self.customers:
            # Incoming edges to c
            incoming_indices: List[int] = []
            for k in range(self.num_vehicles):
                for i in self.nodes:
                    if i == c:
                        continue
                    key = (k, i, c)
                    incoming_indices.append(self.var_index[key])

            self._add_constraint_equality_one(incoming_indices, self.penalty_visit)

            # Outgoing edges from c
            outgoing_indices: List[int] = []
            for k in range(self.num_vehicles):
                for j in self.nodes:
                    if j == c:
                        continue
                    key = (k, c, j)
                    outgoing_indices.append(self.var_index[key])

            self._add_constraint_equality_one(outgoing_indices, self.penalty_visit)

    def _build_depot_constraints(self) -> None:
        """
        Each vehicle must leave the depot exactly once and return exactly once:

        For each vehicle k:
            sum_j x_{k depot j} = 1
            sum_i x_{k i depot} = 1
        """
        d = self.depot
        for k in range(self.num_vehicles):
            # Outgoing from depot for vehicle k
            out_indices: List[int] = []
            for j in self.nodes:
                if j == d:
                    continue
                key = (k, d, j)
                out_indices.append(self.var_index[key])
            self._add_constraint_equality_one(out_indices, self.penalty_depot)

            # Incoming to depot for vehicle k
            in_indices: List[int] = []
            for i in self.nodes:
                if i == d:
                    continue
                key = (k, i, d)
                in_indices.append(self.var_index[key])
            self._add_constraint_equality_one(in_indices, self.penalty_depot)


# ----------------------------------------------------------------------
# Example of basic usage (pseudo-code)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a simple complete graph with distances
    G = nx.complete_graph(5)  # nodes 0, 1, 2, 3, 4
    depot = 0
    for i, j in G.edges():
        G[i][j]["distance"] = np.hypot(i - j, 0)  # dummy distance

    encoder = VRPQUBOEncoder(
        graph=G,
        num_vehicles=2,
        depot=depot,
        distance_attr="distance",
        objective_scale=1.0,
        penalty_visit=10.0,
        penalty_depot=10.0,
    )
    qubo_model = encoder.build_qubo()

    qubo_dict = qubo_model.to_qubo_dict()
    # For PennyLane:
    H = qubo_model.to_hamiltonian()
