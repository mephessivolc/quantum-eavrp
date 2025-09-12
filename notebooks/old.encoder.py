"""encoder.py
=================
QUBO encoder para EA-VRP com **estações de recarga**.

* Variáveis y_{v,g}: veículo *v* atende grupo *g*.
* Variáveis z_{v,r}: veículo *v* visita estação *r* (no máximo uma por veículo).

Disponibiliza utilidades `to_matrix()` e `print_matrix()` para depuração.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from geografics import Distances
from grouping import EAVRPGroup
from utils import Vehicle, Depot

Idx = int
QUBO = Dict[Tuple[Idx, Idx], float]

FAKE_DEPOT = Depot(0, (0,0))

class QUBOEncoder:
    """Codifica EA-VRP + recarga em formato QUBO."""

    def __init__(
        self,
        vehicles: List[Vehicle],
        groups: List[EAVRPGroup],
        stations: List[object] | None,
        penalty_lambda: float | None = None,
        penalty_mu: float | None = None,
        depot: List[Depot] | None = None,
        metric: str = "euclidean",
    ) -> None:
        if not vehicles or not groups:
            raise ValueError("vehicles e groups não podem ser vazios")
        self.vehicles = vehicles
        self.groups = groups
        self.stations = stations or []
        self.metric_fn = getattr(Distances, metric)
        self.depot = depot or FAKE_DEPOT

        # penalidades
        self._lambda_ = penalty_lambda 
        self._mu = penalty_mu or self._lambda_

        # mapeamento variável <-> índice
        self.index_map: Dict[Tuple[str, int, int], Idx] = {}
        idx = 0
        for v in range(len(self.vehicles)):
            for g in range(len(self.groups)):
                self.index_map[("G", v, g)] = idx; idx += 1
            for r in range(len(self.stations)):
                self.index_map[("R", v, r)] = idx; idx += 1
        self.reverse_map = {i: k for k, i in self.index_map.items()}

        # cache do QUBO
        self._Q: QUBO | None = None
        self._offset: float | None = None

    @property
    def num_qubits(self) -> int:
        Q = self.Q  # força build se necessário
        indices = set(i for i, j in Q.keys()) | set(j for i, j in Q.keys())
        return len(indices)

    @property
    def Q(self) -> QUBO:
        if self._Q is None:
            self._Q, self._offset = self._build_qubo()
        return self._Q

    @property
    def offset(self) -> float:
        if self._offset is None:
            _ = self.Q
        return float(self._offset)  # type: ignore
    
    @property
    def lambda_(self) -> float:
        if self._lambda_ is None:
            return 1.0
        return self._lambda_
    
    @property
    def mu(self) -> float:
        if self._mu is None:
            return 1.0
        return self._mu

    def set_penalties(self, lambda_: float, mu: float):
        """
        Atualiza penalidades λ e μ e invalida cache do QUBO.
        """
        self._lambda_ = lambda_
        self._mu = mu
        self._Q = None
        self._offset = None
        return self

    def encode(self):
        """
        Constrói (se necessário) e retorna o QUBO atual:
        (Q, offset, reverse_map)
        """
        if self._Q is None:
            self._Q, self._offset = self._build_qubo()
        return self._Q, self._offset, self.reverse_map

    def to_matrix(self) -> np.ndarray:
        n = self.num_qubits
        mat = np.zeros((n, n))
        for (i, j), c in self.Q.items():
            mat[i, j] += c
            if i != j:
                mat[j, i] += c
        return mat

    def print_matrix(self, precision: int = 2):
        np.set_printoptions(precision=precision, suppress=True)
        print(self.to_matrix())
        np.set_printoptions()

    def _extract_xy(self, obj):
        if isinstance(obj, (tuple, list)):
            return tuple(obj[:2])
        for attr in ("coord", "coords", "location", "pos", "position"):
            xy = getattr(obj, attr, None)
            if xy is not None:
                return tuple(xy[:2])
        raise TypeError("Station sem campo de coordenadas reconhecido")

    def _compute_group_costs(self) -> List[List[float]]:
        metric = self.metric_fn
        depot_coord = self.depot[0].location
        costs = []
        for _ in self.vehicles:
            row = []
            for g in self.groups:
                go, gd = g.origin, g.destination
                cost = (
                    metric(*depot_coord, *go)
                    + g.distance(metric=metric.__name__)
                    + metric(*gd, *depot_coord)
                )
                row.append(cost)
            costs.append(row)
        return costs

    def _compute_station_costs(self) -> List[List[float]]:
        if not self.stations:
            return []
        metric = self.metric_fn
        depot_coord = self.depot[0].location
        costs = []
        for _ in self.vehicles:
            row = []
            for st in self.stations:
                xy = self._extract_xy(st)
                row.append(metric(*depot_coord, *xy) + metric(*xy, *depot_coord))
            costs.append(row)
        return costs

    def _build_qubo(self) -> Tuple[QUBO, float]:
        Q: QUBO = {}
        offset = 0.0

        # custos
        self._cost_g = self._compute_group_costs()
        self._cost_r = self._compute_station_costs()
        # all_costs = [c for row in self._cost_g + self._cost_r for c in row]
        # max_c = max(all_costs) if all_costs else 1.0

        # custos lineares
        for (kind, v, idx2), var in self.index_map.items():
            cost = self._cost_g[v][idx2] if kind == "G" else self._cost_r[v][idx2]
            Q[(var, var)] = Q.get((var, var), 0.0) + cost

        # restrição: cada grupo é atendido exatamente uma vez
        lam, n_v, n_g = self.lambda_, len(self.vehicles), len(self.groups)
        for g in range(n_g):
            vars_g = [self.index_map[("G", v, g)] for v in range(n_v)]
            for var in vars_g:
                Q[(var, var)] = Q.get((var, var), 0.0) - lam
            for i in range(n_v):
                for j in range(i + 1, n_v):
                    vi, vj = vars_g[i], vars_g[j]
                    key = (vi, vj) if vi <= vj else (vj, vi)
                    Q[key] = Q.get(key, 0.0) + 2 * lam
            offset += lam
        
        # restrição: cada veículo atende NO MÁXIMO um grupo
        mu_v = self.mu # ou outro coef. específico
        for v in range(n_v):
            # índices das variáveis y_{v,g}
            vars_v = [self.index_map[("G", v, g)] for g in range(n_g)]

            # —— termos lineares  (-μ y_{vg})
            for var in vars_v:
                Q[(var, var)] = Q.get((var, var), 0.0) - mu_v

            # —— termos quadráticos  (+2μ y_{vg} y_{vg'})
            for i in range(len(vars_v)):
                for j in range(i + 1, len(vars_v)):
                    vi, vj = vars_v[i], vars_v[j]
                    key = (vi, vj) if vi <= vj else (vj, vi)
                    Q[key] = Q.get(key, 0.0) + 2 * mu_v

            # —— termo constante  (+μ)
            offset += mu_v


        # restrição recarga (<=1 por veículo)
        if self.stations:
            mu, n_r = self.mu, len(self.stations)
            for v in range(n_v):
                vars_r = [self.index_map[("R", v, r)] for r in range(n_r)]
                for i in range(n_r):
                    for j in range(i + 1, n_r):
                        vi, vj = vars_r[i], vars_r[j]
                        key = (vi, vj) if vi <= vj else (vj, vi)
                        Q[key] = Q.get(key, 0.0) + 2 * mu

        return Q, offset

    def is_feasible(self, bitstring) -> bool:
        if isinstance(bitstring, str):
            bits = [int(b) for b in bitstring]
        else:
            bits = list(bitstring)

        V, G = len(self.vehicles), len(self.groups)
        y = [bits[i * G : (i + 1) * G] for i in range(V)]
        # cada grupo g exatamente uma vez
        for g in range(G):
            if sum(y[v][g] for v in range(V)) != 1:
                return False
        # cada veículo atende no máximo um grupo
        for v in range(V):
            if sum(y[v][g] for g in range(G)) > 1:
                return False
        return True

    def interpret(self, bitstring):
        """
        Decodifica o bitstring nas escolhas do modelo:
        - y_{v,g} = 1  -> veículo v atende grupo g
        - z_{v,r} = 1  -> veículo v visita estação r

        Retorna um dicionário: {v: {"groups":[g,...], "stations":[r,...]}}
        """
        # garante QUBO construído (reverse_map pronto)
        _ = self.Q

        if isinstance(bitstring, str):
            bits = [int(b) for b in bitstring]
        else:
            bits = list(bitstring)

        assign = {v: {"groups": [], "stations": []} for v in range(len(self.vehicles))}
        for i, b in enumerate(bits):
            if b != 1:
                continue
            try:
                kind, v, idx2 = self.reverse_map[i]
            except KeyError:
                # bit que não mapeia para variável conhecida (deveria não acontecer)
                continue
            if kind == "G":
                assign[v]["groups"].append(idx2)
            elif kind == "R":
                assign[v]["stations"].append(idx2)
        return assign

    def cost(self, bitstring) -> float:
        """
        Custo objetivo *limpo* da solução (sem penalidades), se factível.
        Soma dos custos lineares pré-computados:
          - custo veículo-grupo (ida depósito -> grupo, distância interna do grupo, volta)
          - custo veículo-estação (ida e volta ao depósito).
        """
        # garante que os caches de custo foram construídos
        _ = self.Q
        assign = self.interpret(bitstring)

        # checa factibilidade com as restrições do QUBO
        if not self.is_feasible(bitstring):
            return float("nan")

        total = 0.0
        V = len(self.vehicles)
        for v in range(V):
            # grupos escolhidos
            for g in assign[v]["groups"]:
                total += self._cost_g[v][g]
            # estações escolhidas (se existirem)
            if self.stations:
                for r in assign[v]["stations"]:
                    total += self._cost_r[v][r]
        return float(total)



__all__ = ["QUBOEncoder"]
