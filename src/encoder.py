"""
VRP encoder: VRP instance → QUBO, and bitstring → routes (decoder).
"""

from __future__ import annotations

from typing import Dict, List, Hashable, Optional, Any, Iterable
from pennylane import numpy as np

import networkx as nx

from qubo import QUBO, VarKey


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
        # Example sketch:
        #
        # for (i, j, data) in self.graph.edges(data=True):
        #     dist = float(data.get(self.distance_attr, data.get("weight", 1.0)))
        #     for k in range(self.num_vehicles):
        #         if i != j:
        #             idx = self.var_index[(k, i, j)]
        #             Q[idx, idx] += dist
        #         if j != i:
        #             idx = self.var_index[(k, j, i)]
        #             Q[idx, idx] += dist
        #
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
        #       - self._add_equality_one_constraint(indices, penalty, Q)
        #       - and track constant shifts if needed.
        raise NotImplementedError("Hard constraints encoding not implemented yet.")

    # Here you can add low-level helpers like `_add_qubo_term` and
    # `_add_equality_one_constraint`.
