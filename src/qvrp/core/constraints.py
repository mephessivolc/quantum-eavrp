from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Callable, Optional
import inspect
import networkx as nx

Index = int
NodeId = Any
VehicleId = int
NodeKey = Tuple[VehicleId, NodeId, NodeId]


@dataclass
class ConstraintTerm:
    """
    Quadratic term representing a single penalized constraint.

    linear[i]           -> contribution to c[i]
    quadratic[(i, j)]   -> contribution to Q[i, j]
    offset              -> constant contribution
    """
    name: str
    linear: Dict[Index, float]
    quadratic: Dict[Tuple[Index, Index], float]
    offset: float


class ConstraintCollectorMixin:
    """
    Mixin that automatically collects constraint methods and aggregates
    their ConstraintTerm outputs.

    Strategy:
    - If ACTIVE_CONSTRAINTS is defined in the subclass, use only those methods.
    - Otherwise, auto-collect methods whose name starts with '_constraint_'.
    """

    ACTIVE_CONSTRAINTS: List[str] | None = None

    def get_constraints(
        self,
        graph: nx.Graph,
        var_index: Dict[NodeKey, Index],
        num_vehicles: int,
    ) -> List[ConstraintTerm]:
        """
        Collect and call all constraint-building methods, then aggregate
        their ConstraintTerm outputs.
        """
        methods = self._resolve_constraint_methods()

        all_terms: List[ConstraintTerm] = []
        for method in methods:
            terms = method(
                graph=graph,
                var_index=var_index,
                num_vehicles=num_vehicles,
            )
            all_terms.extend(terms)

        return all_terms

    def _resolve_constraint_methods(self) -> List[Callable]:
        """
        Decide which methods are considered constraint builders.

        - If ACTIVE_CONSTRAINTS is not None, use that list.
        - Otherwise, auto-collect all methods starting with '_constraint_'.
        """
        if self.ACTIVE_CONSTRAINTS is not None:
            methods: List[Callable] = []
            for name in self.ACTIVE_CONSTRAINTS:
                attr = getattr(self, name)
                if callable(attr):
                    methods.append(attr)
            return methods

        # Auto-collect
        methods: List[Callable] = []
        for name, member in inspect.getmembers(self, predicate=callable):
            if name.startswith("_constraint_"):
                methods.append(member)

        # Sort by name for deterministic ordering
        methods.sort(key=lambda m: m.__name__)
        return methods
    
class TaxiConstraints(ConstraintCollectorMixin):
    """
    Base constraint set for a taxi-style VRP (no explicit depot).

    This class focuses on three families of constraints:
    - assignment
    - flow
    - precedence

    The actual QUBO math for each family will be implemented later.
    """

    def __init__(
        self,
        penalty_params: Dict[str, float],
    ) -> None:
        """
        Parameters
        ----------
        penalty_params:
            Dictionary of penalty weights, e.g.:
            - "assignment": weight for assignment constraints
            - "flow": weight for flow constraints
            - "precedence": weight for precedence constraints

        precedence_pairs:
            Optional list of (pickup_node, delivery_node) pairs that
            must respect a precedence relation (pickup before delivery).
        """
        self.penalty_params = penalty_params

        # If you want to explicitly control which constraints are active,
        # you can uncomment and customize this:
        #
        # self.ACTIVE_CONSTRAINTS = [
        #     "_constraint_assignment",
        #     "_constraint_flow",
        #     "_constraint_precedence",
        # ]

    # ==========================
    # ASSIGNMENT
    # ==========================

    def _constraint_assignment(
        self,
        graph: nx.Graph,
        var_index: Dict[Tuple[VehicleId, NodeId, NodeId], Index],
        num_vehicles: int,
    ) -> List[ConstraintTerm]:
        """
        Build assignment-related constraints.

        Current modeling choice
        -----------------------
        For each node k, we enforce that it has exactly one incoming arc
        over all vehicles:

            (sum_{v} sum_{i} x[v, i, k] - 1)^2

        This ensures that each service node k is visited exactly once
        (entry) by some vehicle. Flow constraints (_constraint_flow) will
        later enforce the consistency between entering and leaving nodes.

        QUBO expansion
        --------------
        For a set of binary variables {x_i}:

            (sum_i x_i - 1)^2
            = - sum_i x_i + 2 * sum_{i<j} x_i x_j + 1

        With penalty weight λ = penalty_params["assignment"], this becomes:

            linear coeff for each x_i:      -λ
            quadratic coeff for x_i x_j:    +2λ  (for i<j)
            offset:                         +λ   (per node)

        This method:
        - groups variable indices by destination node k (incoming arcs),
        - builds one ConstraintTerm per node k with non-empty incoming set.
        """
        if "assignment" not in self.penalty_params:
            raise ValueError(
                "Missing 'assignment' penalty in penalty_params for TaxiConstraints."
            )

        lambda_assignment = float(self.penalty_params["assignment"])
        terms: List[ConstraintTerm] = []

        # 1. Agrupar índices por nó de destino k:
        #    node_to_indices[k] = [idx1, idx2, ...] correspondendo a x[v,i,k]
        node_to_indices: Dict[NodeId, List[Index]] = {}

        for (v, i, j), idx in var_index.items():
            # j é o nó de destino
            node_to_indices.setdefault(j, []).append(idx)

        # 2. Para cada nó k com pelo menos uma variável de chegada,
        #    construir o termo de restrição (sum x_i - 1)^2.
        for k, indices in node_to_indices.items():
            if not indices:
                continue  # nada a fazer se não houver variáveis associadas

            linear: Dict[Index, float] = {}
            quadratic: Dict[Tuple[Index, Index], float] = {}
            offset: float = 0.0

            # Coeficientes lineares: -λ para cada x_i
            for idx in indices:
                linear[idx] = linear.get(idx, 0.0) - lambda_assignment

            # Coeficientes quadráticos: +2λ para cada par (i<j)
            n = len(indices)
            for p in range(n):
                i_idx = indices[p]
                for q in range(p + 1, n):
                    j_idx = indices[q]
                    # Garantir (i, j) com i <= j para consistência
                    if i_idx <= j_idx:
                        key = (i_idx, j_idx)
                    else:
                        key = (j_idx, i_idx)
                    quadratic[key] = quadratic.get(key, 0.0) + 2.0 * lambda_assignment

            # Offset constante: +λ
            offset = lambda_assignment

            terms.append(
                ConstraintTerm(
                    name=f"assignment_node_{k}",
                    linear=linear,
                    quadratic=quadratic,
                    offset=offset,
                )
            )

        return terms


    # ==========================
    # FLOW
    # ==========================

    def _constraint_flow(
        self,
        graph: nx.Graph,
        var_index: Dict[NodeKey, Index],
        num_vehicles: int,
    ) -> List[ConstraintTerm]:
        """
        Flow constraint for taxi-style VRP.

        For each vehicle v and node k, we enforce that the number of incoming
        arcs equals the number of outgoing arcs:

            ( sum_i x[v, i, k] - sum_j x[v, k, j] )^2

        Expanding:

            E_flow(v, k) =
                (sum_in x)^2 + (sum_out x)^2 - 2 (sum_in x)(sum_out x)

        This yields:
        - linear terms: +1 * x for every in and out variable,
        - quadratic terms:
            * +2 * x_a x_b for pairs in the "in" set (a != b),
            * +2 * x_a x_b for pairs in the "out" set (a != b),
            * -2 * x_in x_out for cross pairs (in, out).

        All multiplied by the flow penalty parameter.
        """
        penalty = float(self.penalty_params["flow"])
        if penalty == 0.0:
            return []

        terms: List[ConstraintTerm] = []

        nodes = list(graph.nodes())

        for v in range(num_vehicles):
            for k in nodes:
                in_vars: List[Index] = []
                out_vars: List[Index] = []

                # Incoming arcs: (i -> k)
                for i in nodes:
                    if i == k:
                        continue
                    key_in: NodeKey = (v, i, k)
                    if key_in in var_index:
                        in_vars.append(var_index[key_in])

                # Outgoing arcs: (k -> j)
                for j in nodes:
                    if j == k:
                        continue
                    key_out: NodeKey = (v, k, j)
                    if key_out in var_index:
                        out_vars.append(var_index[key_out])

                # If no arcs touch this node for this vehicle, skip
                if not in_vars and not out_vars:
                    continue

                linear: Dict[Index, float] = {}
                quadratic: Dict[Tuple[Index, Index], float] = {}

                # Linear terms: +1 * x for each in and out variable
                for idx in in_vars + out_vars:
                    linear[idx] = linear.get(idx, 0.0) + penalty * 1.0

                # Quadratic terms within incoming set: +2 * x_a x_b
                for a_pos in range(len(in_vars)):
                    for b_pos in range(a_pos + 1, len(in_vars)):
                        i_idx = in_vars[a_pos]
                        j_idx = in_vars[b_pos]
                        if i_idx == j_idx:
                            continue
                        key = (min(i_idx, j_idx), max(i_idx, j_idx))
                        quadratic[key] = quadratic.get(key, 0.0) + penalty * 2.0

                # Quadratic terms within outgoing set: +2 * x_a x_b
                for a_pos in range(len(out_vars)):
                    for b_pos in range(a_pos + 1, len(out_vars)):
                        i_idx = out_vars[a_pos]
                        j_idx = out_vars[b_pos]
                        if i_idx == j_idx:
                            continue
                        key = (min(i_idx, j_idx), max(i_idx, j_idx))
                        quadratic[key] = quadratic.get(key, 0.0) + penalty * 2.0

                # Cross terms (incoming vs outgoing): -2 * x_in x_out
                for i_idx in in_vars:
                    for j_idx in out_vars:
                        if i_idx == j_idx:
                            continue
                        key = (min(i_idx, j_idx), max(i_idx, j_idx))
                        quadratic[key] = quadratic.get(key, 0.0) - penalty * 2.0

                # Offset: no constant term in (sum_in - sum_out)^2
                offset = 0.0

                name = f"flow_v{v}_node_{k}"
                terms.append(
                    ConstraintTerm(
                        name=name,
                        linear=linear,
                        quadratic=quadratic,
                        offset=offset,
                    )
                )

        return terms
