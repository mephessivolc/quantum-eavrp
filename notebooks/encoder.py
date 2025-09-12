"""encoder.py — PyQUBO (objetivo puro)
=================================================
Formula o objetivo do EA‑VRP com **grupos como commodities** usando PyQUBO.

Escopo desta versão
-------------------
- Apenas **função objetivo**: minimizar a distância para que um veículo leve
  um **grupo** da sua origem ao destino, incluindo ida/volta ao depósito.
- **Sem** restrições nem penalidades ainda. A API expõe pontos de extensão
  para inserir penalidades posteriormente.
- Paradigma **POO** e compatível com os *solvers* do projeto:
  - `encode() -> (Q, offset, reverse_map)`
  - `to_matrix()`
  - `num_qubits`, `Q`, `offset`
  - `interpret(bits)` e `is_feasible(bits)` (triviais por ora)

Convenções
----------
- Variáveis binárias: **y_{v,g}** ≡ veículo *v* atende grupo *g*.
- Custo linear c_{v,g}:
    depot→origem(g)  +  custo interno do grupo  +  destino(g)→depot
  O depósito usado no custo é aquele que **minimiza** (ida+volta) para o par (v,g).
- O grafo é **não direcionado** e a métrica é obtida em `geografics.Distances`.

Extensões futuras (ganchos prontos)
-----------------------------------
- Métodos `add_penalty(...)` e `build_hamiltonian()` para anexar termos
  quadráticos (restrições) sem reescrever o encoder.
- Espaço reservado para custos envolvendo **estações de recarga**.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np

from pyqubo import Binary

# Projeto
from geografics import Distances
from utils import Vehicle, Depot
from grouping import EAVRPGroup as Group

Idx = int
QuboDict = Dict[Tuple[Idx, Idx], float]


class Encoder:
    """Encoder QUBO baseado em PyQUBO apenas com a **função objetivo**.

    Parâmetros
    ----------
    vehicles : List[Vehicle]
    groups   : List[Group]
    recharge_points : List[Any] | None
        Mantido para compatibilidade; não utilizado no objetivo puro.
    depots   : List[Depot]
    metric   : str
        Nome da função em `Distances` (e.g. 'euclidean', 'haversine').
    scale    : float
        Fator de escala opcional aplicado aos custos.
    """

    def __init__(
        self,
        vehicles: List[Vehicle],
        groups: List[Group],
        recharge_points: List[Any] | None,
        depot: List[Depot] | None,
        metric: str = "euclidean",
        scale: float = 1.0,
    ) -> None:
        if not vehicles or not groups:
            raise ValueError("vehicles e groups não podem ser vazios")
        if not depot:
            raise ValueError("é necessário ao menos um depósito")

        self.vehicles = vehicles
        self.groups = groups
        self.stations = recharge_points or []
        self.depots = depot
        self.metric_fn = getattr(Distances, metric)
        self.scale = float(scale)

        # --- caches internos ---
        self._cost: List[List[float]] | None = None   # c_{v,g}
        self._symbolic = None                         # expressão simbólica PyQUBO
        self._compiled = None                         # modelo compilado
        self._Q: QuboDict | None = None
        self._offset: float | None = None

        # mapeamento variável ↔ índice (ordem estável)
        self.index_map: Dict[Tuple[int, int], Idx] = {}
        self.reverse_map: Dict[Idx, Tuple[int, int]] = {}
        self._build_variable_indexing()

    # ------------------------------------------------------------------
    # Preparação
    # ------------------------------------------------------------------
    def _build_variable_indexing(self) -> None:
        idx = 0
        for v in range(len(self.vehicles)):
            for g in range(len(self.groups)):
                self.index_map[(v, g)] = idx
                self.reverse_map[idx] = (v, g)
                idx += 1

    def _best_depot_cost(self, v: Vehicle, g: Group) -> float:
        """Escolhe o depósito que minimiza ida+volta para o par (v,g).
        Retorna: min_d [ d→origem(g) + destino(g)→d ].
        """
        best = float("inf")
        for d in self.depots:
            dep = d.location
            go = g.origin  # centróide de origens
            gd = g.destination  # centróide de destinos
            val = self.metric_fn(*dep, *go) + self.metric_fn(*gd, *dep)
            if val < best:
                best = val
        return best

    def _internal_group_distance(self, g: Group) -> float:
        """Custo interno de atender todos os passageiros do grupo.
        Usa a heurística gulosa definida em EAVRPGroup.distance().
        """
        # EAVRPGroup.distance aceita 'metric' como str; padronizamos
        return float(g.distance(metric=self.metric_fn.__name__))

    def _compute_cost_matrix(self) -> List[List[float]]:
        """Constroi c_{v,g} para todos veículos e grupos."""
        if self._cost is not None:
            return self._cost
        costs: List[List[float]] = []
        for v in self.vehicles:
            row: List[float] = []
            for g in self.groups:
                base = self._best_depot_cost(v, g)
                internal = self._internal_group_distance(g)
                row.append(self.scale * (base + internal))
            costs.append(row)
        self._cost = costs
        return costs

    # ------------------------------------------------------------------
    # Construção do Hamiltoniano (objetivo puro)
    # ------------------------------------------------------------------
    def build_symbolic(self):
        """Cria a expressão simbólica PyQUBO do objetivo: sum c_{v,g} * y_{v,g}."""
        if self._symbolic is not None:
            return self._symbolic

        costs = self._compute_cost_matrix()
        expr = 0
        # cria/guarda as variáveis binárias na mesma ordem do index_map
        self._y_vars: Dict[Tuple[int, int], Any] = {}
        for (v, g), _idx in self.index_map.items():
            y = Binary(f"y_{v}_{g}")
            self._y_vars[(v, g)] = y
            expr += costs[v][g] * y

        self._symbolic = expr
        return expr

    def build_hamiltonian(self):
        """Compila a expressão simbólica para QUBO (sem penalidades)."""
        if self._compiled is not None:
            return self._compiled
        expr = self.build_symbolic()
        self._compiled = expr.compile()
        return self._compiled

    # ------------------------------------------------------------------
    # API pública compatível
    # ------------------------------------------------------------------
    @property
    def Q(self) -> QuboDict:
        if self._Q is None:
            self._materialize_qubo()
        return self._Q  # type: ignore

    @property
    def offset(self) -> float:
        if self._offset is None:
            self._materialize_qubo()
        return float(self._offset)  # type: ignore

    @property
    def num_qubits(self) -> int:
        return len(self.index_map)

    def encode(self) -> Tuple[QuboDict, float, Dict[int, Tuple[int, int]]]:
        """Retorna (Q, offset, reverse_map)."""
        return self.Q, self.offset, self.reverse_map

    def to_matrix(self) -> np.ndarray:
        n = self.num_qubits
        mat = np.zeros((n, n))
        for (i, j), c in self.Q.items():
            mat[i, j] += c
            if i != j:
                mat[j, i] += c
        return mat

    def print_matrix(self, precision: int = 3) -> None:
        np.set_printoptions(precision=precision, suppress=True)
        print(self.to_matrix())
        np.set_printoptions()

    def interpret(self, bitstring: List[int] | str) -> Dict[int, List[int]]:
        """Mapeia bits ativos para {veiculo: [grupos...]}."""
        if isinstance(bitstring, str):
            bits = [int(b) for b in bitstring]
        else:
            bits = list(map(int, bitstring))
        assign: Dict[int, List[int]] = {}
        for i, b in enumerate(bits):
            if b != 1:
                continue
            v, g = self.reverse_map.get(i, (-1, -1))
            assign.setdefault(v, []).append(g)
        return assign

    def pretty_objective(self, latex: bool = True) -> str:
        """
        Retorna a função objetivo em forma legível.
        latex=True → expressão LaTeX (Σ c_{v,g} y_{v,g})
        latex=False → expressão textual simples
        """
        costs = self._compute_cost_matrix()
        terms = []
        for v in range(len(self.vehicles)):
            for g in range(len(self.groups)):
                coef = costs[v][g]
                if coef == 0:
                    continue
                if latex:
                    terms.append(f"{coef:.2f} \\times y_{{{v},{g}}}")
                else:
                    terms.append(f"{coef:.2f}*y_{v}_{g}")
        sep = " + " if not latex else " + "
        expr = sep.join(terms)
        if latex:
            return f"$\\min {expr}$"
        return f"min {expr}"


    def is_feasible(self, _bitstring: List[int] | str) -> bool:
        """Sem restrições por ora → sempre True."""
        return True

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _materialize_qubo(self) -> None:
        compiled = self.build_hamiltonian()
        qubo, offset = compiled.to_qubo()  # feed_dict vazio; sem Placeholders

        # Reindexa QUBO no **mesmo** ordenamento de self.index_map
        # PyQUBO retorna chaves como ("y_0_0", "y_1_2"), etc. Precisamos
        # traduzi-las para índices inteiros consistentes com reverse_map.
        name_to_idx: Dict[str, int] = {
            f"y_{v}_{g}": idx for (v, g), idx in self.index_map.items()
        }
        Q: QuboDict = {}
        for (a, b), w in qubo.items():
            i = name_to_idx[a]
            j = name_to_idx[b]
            key = (i, j) if i <= j else (j, i)
            Q[key] = Q.get(key, 0.0) + float(w)

        self._Q = Q
        self._offset = float(offset)


__all__ = ["Encoder"]


if __name__ == "__main__":  # autoteste rápido
    from feasible_instance_generator import InstanceGenerator
    from grouping import GeoGrouper

    # pequena instância
    ig = InstanceGenerator(n_passengers=6, n_vehicles=3, n_depots=1, n_recharges=1, seed=0)
    V, P, D, R = ig.build()
    groups = GeoGrouper(max_size=3, delta=0.004).fit(P)

    enc = Encoder(V, groups, R, D, metric="euclidean", scale=100.0)
    Q, off, rmap = enc.encode()
    print(f"#qubits = {enc.num_qubits}")
    print(f"offset  = {off:.3f}")
    enc.print_matrix(2)
    # exemplo de interpretação
    sample = [0]*enc.num_qubits
    if sample:
        sample[0] = 1
    print("interpret:", enc.interpret(sample))

    print()
    print(enc.pretty_objective(latex=False))   # min 12.30*y_0_0 + 8.50*y_0_1 ...
    print(enc.pretty_objective(latex=True))  
