# decoder.py

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Iterable
import networkx as nx

from pennylane import numpy as np

from qvrp.core.constraints import ConstraintTerm 

Index = int
NodeId = Any
VehicleId = int
NodeKey = Tuple[Any, Any, Any]  # (vehicle_id, origin_node, destination_node)


@dataclass
class VehicleRoute:
    """
    Decoded route for a single vehicle.

    Attributes
    ----------
    vehicle_id:
        Identifier of the vehicle.
    nodes:
        Ordered list of node ids visited by this vehicle.
        For an open taxi-style route, this is typically:
            [start_node, ..., end_node]
    cost:
        Total travel cost of this route, according to the graph's edge
        attributes (e.g. distance, time).
    """

    vehicle_id: VehicleId
    nodes: List[NodeId]
    cost: float


@dataclass
class DecodedSolution:
    """
    Container for a fully decoded VRP solution.

    Attributes
    ----------
    routes:
        List of per-vehicle routes. Vehicles with empty routes may or may
        not be included, depending on the decoding policy.
    total_cost:
        Sum of all route costs.
    is_feasible:
        Boolean flag indicating whether the solution satisfies the VRP
        constraints under the chosen feasibility checks.
    violated_constraints:
        Optional dictionary describing which constraints are violated and
        by how much (e.g. counts, magnitudes). The exact format can be
        refined later.
    raw_bitstring:
        The original bitstring used for decoding, stored for traceability.
    """

    routes: List[VehicleRoute]
    total_cost: float
    is_feasible: bool
    violated_constraints: Dict[str, Any]
    raw_bitstring: List[int]


class Decoder:
    """
    Decoder for taxi-style VRP solutions encoded as QUBO bitstrings.

    Responsibilities
    ----------------
    - Interpret a binary solution vector (bitstring) using the same
      variable index mapping as the Encoder: (vehicle, origin, dest) -> idx.
    - Extract active arcs for each vehicle from the bitstring.
    - Reconstruct ordered vehicle routes from these arcs.
    - Compute route and total costs using the underlying graph.
    - Optionally, check basic feasibility (flow, assignment, precedence),
      possibly in cooperation with a constraints handler.

    This class is intentionally backend-agnostic: it does not care
    whether the bitstring came from a quantum solver, classical solver,
    or brute-force enumeration.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_vehicles: int,
        var_index: Dict[NodeKey, Index],
        cost_attribute: str = "weight",
        constraint_terms: Optional[List[ConstraintTerm]] = None,
    ) -> None:
        """
        Initialize the decoder with a fixed problem instance.

        Parameters
        ----------
        graph:
            NetworkX graph representing the same instance used by the Encoder.
            A copy is stored internally to avoid external mutations.
        num_vehicles:
            Number of vehicles in the instance.
        var_index:
            Variable index mapping produced by the Encoder:
                (vehicle_id, origin_node, destination_node) -> bit index.
        cost_attribute:
            Name of the edge attribute storing the travel cost to be
            used when aggregating route and total costs.
        constraint_terms:
            List of ConstraintTerm objects used in the Encoder to build the QUBO.
            They will be reused here to measure constraint violations on a
            given bitstring, avoiding duplicated logic.
        """
        self.graph: nx.Graph = graph.copy()
        self.num_vehicles: int = num_vehicles
        self.var_index: Dict[NodeKey, Index] = dict(var_index)
        self.cost_attribute: str = cost_attribute
        self.constraint_terms: List[ConstraintTerm] = constraint_terms or []

        # Inverted index: bit index -> (vehicle_id, origin_node, destination_node)
        self._index_to_key: Dict[Index, NodeKey] = {
            idx: key for key, idx in self.var_index.items()
        }

    # ==========================
    # PUBLIC ENTRY POINT
    # ==========================

    def decode(self, bitstring: Iterable[int]) -> DecodedSolution:
        """
        Decode a binary solution vector into a VRP solution in route form.

        Steps:
        ------
        1. Normalize the bitstring into a numpy array x ∈ {0,1}^n.
        2. Extract active arcs for each vehicle from x.
        3. Reconstruct routes for each vehicle from its active arcs.
        4. Compute per-route and total costs using the graph.
        5. Evaluate feasibility and collect violation diagnostics.
        6. Pack everything into a DecodedSolution object.

        Parameters
        ----------
        bitstring:
            Iterable of 0/1 values (or booleans) with length equal to the
            number of variables in var_index.

        Returns
        -------
        DecodedSolution:
            Structured representation of the decoded solution.
        """
        # 1. Normalize bitstring
        x = self._normalize_bitstring(bitstring)

        # 2. Extract active arcs grouped by vehicle
        arcs_by_vehicle = self._extract_active_arcs(x)

        # 3. Build routes and compute route costs
        routes: List[VehicleRoute] = []

        for v in range(self.num_vehicles):
            vehicle_arcs = arcs_by_vehicle.get(v, [])

            # _build_routes_for_vehicle may return 0, 1 or multiple paths
            paths_for_v = self._build_routes_for_vehicle(
                vehicle_id=v,
                arcs=vehicle_arcs,
            )

            for path_nodes in paths_for_v:
                cost = self._compute_route_cost(path_nodes)
                routes.append(
                    VehicleRoute(
                        vehicle_id=v,
                        nodes=path_nodes,
                        cost=cost,
                    )
                )

        # 4. Aggregate total cost
        total_cost = float(sum(route.cost for route in routes))

        # 5. Feasibility and diagnoses
        is_feasible, violations = self._check_feasibility(x=x, routes=routes)

        # 6. Pack result
        solution = DecodedSolution(
            routes=routes,
            total_cost=total_cost,
            is_feasible=is_feasible,
            violated_constraints=violations,
            raw_bitstring=[int(val) for val in x],
        )

        return solution

    # ==========================
    # STEP 1: PARSE BITSTRING
    # ==========================

    def _normalize_bitstring(self, bitstring: Iterable[int]) -> np.ndarray:
        """
        Convert the input bitstring into a numpy array of shape (n,)
        with values in {0, 1}.

        Responsibilities
        ----------------
        - Ensure the length matches the number of variables implied by
          var_index (i.e., number of indices in self._index_to_key).
        - Ensure that all entries are 0 or 1.
        - Provide a consistent internal representation for subsequent
          decoding steps.
        """
        # Convert to numpy array of integers
        x = np.array(list(bitstring), dtype=int)

        n_vars = len(self._index_to_key)

        # Check length consistency
        if x.shape[0] != n_vars:
            raise ValueError(
                f"Bitstring length {x.shape[0]} does not match number of "
                f"variables {n_vars} implied by var_index."
            )

        # Check that all entries are 0 or 1
        unique_vals = set(int(v) for v in np.unique(x))
        allowed = {0, 1}

        if not set(unique_vals).issubset(allowed):
            raise ValueError(
                f"Bitstring contains values outside {{0,1}}: {unique_vals}."
            )

        return x

    def _extract_active_arcs(
        self,
        x: np.ndarray,
    ) -> Dict[VehicleId, List[Tuple[NodeId, NodeId]]]:
        """
        Extract active arcs for each vehicle from the binary vector x.

        Parameters
        ----------
        x:
            Numpy array of shape (n,) with values 0 or 1 representing
            the chosen solution.

        Returns
        -------
        arcs_by_vehicle:
            Dictionary mapping vehicle_id -> list of (origin_node, dest_node)
            pairs corresponding to arcs with x[idx] = 1.

        Notes
        -----
        - This method only interprets the bitstring using var_index; it does
          not attempt to order the arcs into routes or validate feasibility.
        """
        if x.shape[0] != len(self._index_to_key):
            raise ValueError(
                "Bitstring length does not match the number of indexed variables."
            )

        arcs_by_vehicle: Dict[VehicleId, List[Tuple[NodeId, NodeId]]] = {}

        # Iterate over all indices; whenever x[idx] == 1, interpret that
        # index as an active arc (v, i, j).
        for idx, val in enumerate(x):
            if val == 0:
                continue

            # Retrieve semantic key for this variable
            try:
                vehicle_id, origin_node, dest_node = self._index_to_key[idx]
            except KeyError:
                raise KeyError(
                    f"Bit index {idx} not found in index-to-key mapping. "
                    "Decoder var_index and bitstring may be inconsistent."
                )

            arcs_by_vehicle.setdefault(vehicle_id, []).append(
                (origin_node, dest_node)
            )

        return arcs_by_vehicle

    # ==========================
    # STEP 2: BUILD ROUTES
    # ==========================

    def _compute_route_cost(
        self,
        nodes: List[NodeId],
    ) -> float:
        """
        Compute the cost of a route based on the underlying graph.

        Parameters
        ----------
        nodes:
            Ordered list of nodes representing the route:
                [n0, n1, ..., nk]

        Returns
        -------
        cost:
            Sum of the edge costs for consecutive pairs (n0,n1), (n1,n2), ...,
            using the graph's edge attribute given by self.cost_attribute.

        Raises
        ------
        KeyError:
            If an edge in the route does not exist in the graph or does not
            contain the required cost attribute.
        """
        # Trivial case: route with 0 or 1 node has zero travel cost
        if len(nodes) < 2:
            return 0.0

        total_cost = 0.0

        # Sum cost over each consecutive pair (u, v)
        for u, v in zip(nodes, nodes[1:]):
            if not self.graph.has_edge(u, v):
                raise KeyError(
                    f"Edge ({u}, {v}) not found in graph when computing route cost."
                )

            edge_data = self.graph[u][v]

            if self.cost_attribute not in edge_data:
                raise KeyError(
                    f"Edge ({u}, {v}) has no attribute "
                    f"'{self.cost_attribute}' for route cost computation."
                )

            total_cost += float(edge_data[self.cost_attribute])

        return total_cost


    # ==========================
    # STEP 3: FEASIBILITY CHECKS
    # ==========================

    def _check_feasibility(
        self,
        x: np.ndarray,
        routes: List[VehicleRoute],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check feasibility based on the constraint terms evaluated at x.

        Strategy
        --------
        - If no constraint_terms were provided, we consider the solution
          feasible by default.
        - Otherwise:
            * evaluate all constraint energies on x,
            * any constraint with energy > tolerance is considered violated.

        Notes
        -----
        - routes are currently not used in the formal feasibility decision,
          but they are available here if you later want to enrich diagnostics
          (e.g., map violation names to specific nodes or vehicles).
        """
        if not self.constraint_terms:
            return True, {}

        energies = self._evaluate_all_constraints(x)

        tol = 1e-6  # tolerância numérica
        violations: Dict[str, float] = {
            name: e for name, e in energies.items() if e > tol
        }

        is_feasible = len(violations) == 0
        return is_feasible, violations

    def _evaluate_constraint_term(self, term: ConstraintTerm, x: np.ndarray) -> float:
        """
        Evaluate a single constraint term on a given bitstring x.

        E_term(x) = sum_{i,j} Q_ij x_i x_j + sum_i c_i x_i + offset

        Here we avoid building a full matrix; we use the sparse dictionaries
        term.quadratic and term.linear.
        """
        quad = 0.0
        for (i, j), coeff in term.quadratic.items():
            quad += coeff * x[i] * x[j]

        linear = 0.0
        for i, coeff in term.linear.items():
            linear += coeff * x[i]

        return quad + linear + term.offset

    def _evaluate_all_constraints(self, x: np.ndarray) -> Dict[str, float]:
        """
        Evaluate all constraint terms on the bitstring x.

        Returns
        -------
        energies:
            Dict mapping constraint name -> energy value of that constraint.
        """
        energies: Dict[str, float] = {}

        for term in self.constraint_terms:
            e = self._evaluate_constraint_term(term, x)
            energies[term.name] = e

        return energies


    def is_feasible(
        self,
        routes: List[VehicleRoute],
        bitstring: Iterable[int],
    ) -> bool:
        """
        Public, boolean-only feasibility check.

        Parameters
        ----------
        routes:
            List of per-vehicle routes.

        Returns
        -------
        bool:
            True if the solution is considered feasible under the current
            feasibility rules, False otherwise.
        """
        x = self._normalize_bitstring(bitstring)
        is_ok, _ = self._check_feasibility(x=x, routes=routes)

        return is_ok

    def check_diagnoses(
        self,
        routes: List[VehicleRoute],
        bitstring: Iterable[int],
    ) -> Dict[str, float]:
        """
        Public diagnostic interface for constraint violations.

        Parameters
        ----------
        routes:
            List of per-vehicle routes.

        Returns
        -------
        violations:
            Dictionary with detailed information about violated constraints.
            The exact structure can be refined later, but may include e.g.:

            {
                "assignment": {"violated_nodes": [...], "count": int, ...},
                "flow": {"broken_paths": [...], ...},
                "precedence": {"pairs_violated": [...], ...},
            }
        """
        x = self._normalize_bitstring(bitstring)
        _, violations = self._check_feasibility(x=x, routes=routes)

        return violations
    
    def _build_routes_for_vehicle(
        self,
        vehicle_id: VehicleId,
        arcs: List[Tuple[NodeId, NodeId]],
    ) -> List[List[NodeId]]:
        """
        Reconstruct one or more ordered routes from a list of arcs for a
        single vehicle.

        Strategy
        --------
        - Consider the directed graph defined only by the arcs of this vehicle.
        - Build adjacency (sucessores) e graus de entrada/saída.
        - Enquanto houver arestas não utilizadas:
            * escolher um nó de partida:
                - preferir nós com in_degree == 0 e out_degree > 0 (início de caminho),
                - se não existir, escolher qualquer nó com arestas restantes
                  (caso típico: ciclo).
            * seguir sucessores consumindo cada aresta uma única vez, gerando
              um caminho [n0, n1, ..., nk].
        - Retornar a lista de caminhos (rotas) para esse veículo.

        Observações
        -----------
        - Em uma solução viável, espera-se tipicamente 0 ou 1 rota simples
          por veículo, sem ramificações.
        - Se houver múltiplas componentes (vários caminhos) ou ciclos, eles
          serão retornados separadamente; a viabilidade será tratada depois
          por _check_feasibility().
        """
        if not arcs:
            return []

        # Construir graus e adjacência
        in_deg: Dict[NodeId, int] = defaultdict(int)
        out_deg: Dict[NodeId, int] = defaultdict(int)
        succ: Dict[NodeId, List[NodeId]] = defaultdict(list)
        nodes: set[NodeId] = set()

        for origin, dest in arcs:
            succ[origin].append(dest)
            out_deg[origin] += 1
            in_deg[dest] += 1
            nodes.add(origin)
            nodes.add(dest)

        # Cópia local da adjacência para consumir arestas conforme forem usadas
        succ_local: Dict[NodeId, List[NodeId]] = {
            u: list(vs) for u, vs in succ.items()
        }

        def pick_start_node() -> Optional[NodeId]:
            """
            Escolhe um nó de partida para iniciar um caminho:

            - Preferir nó com out_deg > 0 e in_deg == 0 (início natural de caminho).
            - Se não houver, escolher qualquer nó com arestas remanescentes
              (caso típico de ciclos).
            """
            candidates = [n for n in nodes if succ_local.get(n)]

            if not candidates:
                return None

            for n in candidates:
                if in_deg[n] == 0 and out_deg[n] > 0:
                    return n

            # Nenhum nó com in_deg == 0: provavelmente ciclo ou estrutura irregular
            return candidates[0]

        routes: List[List[NodeId]] = []

        while True:
            start = pick_start_node()
            if start is None:
                break

            path: List[NodeId] = [start]
            current = start

            # Caminha consumindo arestas sucessivas até não haver mais saída
            while succ_local.get(current):
                next_node = succ_local[current].pop(0)
                path.append(next_node)
                current = next_node

            routes.append(path)

        return routes