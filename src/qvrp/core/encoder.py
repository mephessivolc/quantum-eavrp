# encoder.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
from pennylane import numpy as np
import networkx as nx

from qvrp.core.constraints import ConstraintTerm

Index = int
NodeId = Any
VehicleId = int

class Encoder:
    """
    QUBO encoder for a taxi-style VRP defined on a NetworkX graph.

    This class is intentionally minimal. It:
    - stores a copy of the graph,
    - defines the variable indexing structure,
    - allocates Q, c, offset,
    - exposes hooks to add objective and constraint terms,
    - returns (Q, c, offset, var_index) for downstream quantum solvers.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_vehicles: int,
        penalty_params: Optional[Dict[str, float]] = None,
        constraints_handler: Optional[Any] = None,
        cost_attribute: str = "distance",
    ) -> None:
        """
        Initialize the encoder with a fixed VRP instance.

        Parameters
        ----------
        graph:
            NetworkX graph representing the instance. A copy is stored
            internally to avoid accidental external mutations.
        num_vehicles:
            Number of vehicles available.
        penalty_params:
            Penalty weights for constraints.
        constraints_handler:
            Optional object responsible for building constraint terms.
            (We will define its interface later.)
        """
        # Problem data
        self.graph: nx.Graph = graph.copy()
        self.num_vehicles: int = num_vehicles
        self.penalty_params: Dict[str, float] = penalty_params or {}
        self.constraints_handler: Optional[Any] = constraints_handler
        self.cost_attribute: str = cost_attribute

        # Variable indexing: (vehicle_id, origin_node, destination_node) -> bit index
        self._var_index: Dict[Tuple[VehicleId, NodeId, NodeId], Index] = {}
        self._num_vars: int = 0

        # QUBO components (allocated after variable index is built)
        self.Q: Optional[np.ndarray] = None
        self.c: Optional[np.ndarray] = None
        self.offset: float = 0.0

    # ==========================
    # PUBLIC MAIN ENTRY POINT
    # ==========================

    def encode(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[Tuple[VehicleId, NodeId, NodeId], Index]]:
        """
        Orchestrate the full encoding process.

        High-level steps:
        1. Build variable index (mapping VRP decisions to bit indices).
        2. Initialize Q, c, and offset.
        3. Add base travel cost terms.
        4. Add additional objective terms (optional).
        5. Add constraint terms (via constraints_handler or custom logic).
        6. Finalize and return Q, c, offset, var_index.
        """
        self._build_variable_index()
        self._initialize_qubo_matrices()
        self._add_travel_cost_terms()
        self._add_additional_objective_terms()
        self._add_constraint_terms()
        return self._finalize_qubo()

    # ==========================
    # STEP 1: VARIABLES / INDICES
    # ==========================

    def _build_variable_index(self) -> None:
        """
        Define which decisions become binary variables and assign indices.

        Typical pattern for an arc-based taxi-style model:
        - For each vehicle v
        - For each pair of distinct nodes (i, j)
        - Create a variable x[v, i, j] indicating that v travels directly i -> j (i != j).

        This method must:
        - fill self._var_index with keys (v, i, j) mapped to integer indices,
        - update self._num_vars with the total number of variables.
        """
        # To be implemented
        self._var_index.clear()
        current_index: int = 0

        # We fix an iteration order over nodes to ensure determinism.
        # If nodes are sortable (e.g. int, str), sorted(self.graph.nodes)
        # gives a stable order. If not, list(self.graph.nodes) preserves
        # the graph's insertion order.
        try:
            nodes = sorted(self.graph.nodes)
        except TypeError:
            nodes = list(self.graph.nodes)

        for v in range(self.num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue  # no self-loop variable for now

                    key = (v, i, j)

                    # Defensive: avoid accidental duplicates
                    if key in self._var_index:
                        # In this basic design, this should never happen.
                        # If it does, it's a modeling bug.
                        raise ValueError(f"Duplicate variable key detected: {key}")

                    self._var_index[key] = current_index
                    current_index += 1

        self._num_vars = current_index

    # ==========================
    # STEP 2: INITIALIZATION
    # ==========================

    def _initialize_qubo_matrices(self) -> None:
        """
        Allocate Q, c, and offset based on self._num_vars.

        After _build_variable_index has been executed, self._num_vars must
        be known. Then:
        - Q is an (N x N) zero matrix (quadratic terms),
        - c is a length-N zero vector (linear terms),
        - offset is a scalar (constant term, already initialized to 0.0).
        """
        if self._num_vars <= 0:
            raise ValueError(
                "Number of variables is zero or negative; "
                "did you run _build_variable_index() correctly?"
            )

        n = self._num_vars

        # Use float64 by default; it is a good compromise between precision
        # and compatibility with most numerical/quantum toolchains.
        self.Q = np.zeros((n, n), dtype=float)
        self.c = np.zeros(n, dtype=float)
        self.offset = 0.0


    # ==========================
    # STEP 3: OBJECTIVE FUNCTION
    # ==========================

    def _add_travel_cost_terms(self) -> None:
        """
        Add base travel cost terms to Q and/or c.

        In a simple formulation:
        - For each variable x[v, i, j], read the cost of arc (i, j)
          from self.graph (e.g., edge attribute "distance" or "time"),
        - convert this into linear contributions on self.c[idx]
          (and possibly quadratic terms on self.Q if needed).

        This method only deals with the pure travel cost, without constraints.
        """
        if self.c is None or self.Q is None:
            raise RuntimeError(
                "QUBO matrices are not initialized. "
                "Call _initialize_qubo_matrices() before _add_travel_cost_terms()."
            )

        for (v, i, j), idx in self._var_index.items():
            # We assume the graph is (at least) undirected, so (i,j) and (j,i)
            # share the same edge data in a nx.Graph.
            if self.graph.has_edge(i, j):
                edge_data = self.graph[i][j]
                cost = edge_data.get(self.cost_attribute, 0.0)
            else:
                # If for some reason there is no edge, we treat the cost as 0.0.
                # Later podemos mudar isso para lançar erro, se fizer sentido.
                cost = 0.0

            # Linear term: cost * x[v,i,j]
            self.c[idx] += float(cost)

    def _add_additional_objective_terms(self) -> None:
        """
        Optional hook to add extra objective components.

        Examples:
        - penalizing long idle times,
        - rewarding balanced usage of vehicles,
        - adding soft preferences.

        By default, this method does nothing.
        Subclasses may override it if needed.
        """
        # Default: no extra objective terms
        pass

    # ==========================
    # STEP 4: CONSTRAINT TERMS
    # ==========================

    def _add_constraint_terms(self) -> None:
        """
        Add constraint-based contributions to Q, c, and offset.

        Typical usage:
        - If constraints_handler is provided, call something like:
          terms = constraints_handler.get_constraints(self.graph, self._var_index, ...)
          and accumulate these terms into self.Q, self.c, self.offset.

        - Alternatively, implement constraint penalties directly here,
          especially for small or prototype models.

        This method is a central hook where all constraints are applied
        on top of the base travel cost.
        """
        if self.constraints_handler is None:
            return

        if self.Q is None or self.c is None:
            raise RuntimeError(
                "QUBO matrices are not initialized. "
                "Call _initialize_qubo_matrices() before _add_constraint_terms()."
            )

        terms: list[ConstraintTerm] = self.constraints_handler.get_constraints(
            graph=self.graph,
            var_index=self._var_index,
            num_vehicles=self.num_vehicles,
        )

        for term in terms:
            # Quadratic contributions
            for (i, j), coeff in term.quadratic.items():
                self.Q[i, j] += coeff
                # Opcional: deixar simetrização para _finalize_qubo
                # Se preferir garantir simetria aqui:
                # if i != j:
                #     self.Q[j, i] += coeff

            # Linear contributions
            for i, coeff in term.linear.items():
                self.c[i] += coeff

            # Constant contribution
            self.offset += term.offset


    # ==========================
    # STEP 5: FINALIZATION
    # ==========================

    def _finalize_qubo(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[Tuple[VehicleId, NodeId, NodeId], Index]]:
        """
        Final adjustments and return Q, c, offset, var_index.

        Typical tasks:
        - ensure self.Q is symmetric (if needed),
        - clean tiny numerical noise (optional),
        - cast Q and c to desired dtypes (e.g., float64).

        Returns
        -------
        Q:
            Quadratic coefficient matrix (N x N).
        c:
            Linear coefficient vector (length N).
        offset:
            Constant term of the QUBO.
        var_index:
            Mapping from (vehicle_id, origin_node, destination_node) to bit index.
        """
        if self.Q is None or self.c is None:
            raise RuntimeError("QUBO matrices are not initialized. Call encode() first.")

        # Ensure symmetry (in case constraint terms only filled one triangle)
        self.Q = 0.5 * (self.Q + self.Q.T)

        return self.Q, self.c, self.offset, dict(self._var_index)