"""grouping.py
================
Heurística gulosa de agrupamento geográfico com restrição de capacidade
(1 a max_size passageiros) e diâmetro máximo δ.

Classes
-------
- GeoGrouper: implementa o método fit(passengers) retornando lista de EAVRPGroup.
- EAVRPGroup: agrupa passageiros e oferece centróide e distância interna.
"""

from __future__ import annotations

from typing import List, Tuple
import math
import numpy as np

from geografics import Distances
from utils import Passenger, Group

Coord = Tuple[float, float]

class EAVRPGroup(Group):

    @property
    def origin(self) -> Coord:
        """
        Calcula o ponto (ilustrativo) de origem do grupo como o centróide
        das coordenadas de origem de todos os passageiros, usando numpy.
        """
        coord = np.array([p.origin for p in self.passengers])

        centroid = coord.mean(axis=0)
        return tuple(centroid)

    @property
    def destination(self) -> Coord:
        """
        Calcula o ponto (ilustrativo) de destino do grupo como o centróide
        das coordenadas de destino de todos os passageiros, usando numpy.
        """
        coord = np.array([p.destination for p in self.passengers])

        centroid = coord.mean(axis=0)
        return tuple(centroid)
    
    def distance(self, 
                 metric: str="euclidean", 
                 depot: Tuple[float,float]=(0,0)
                 ):   

        """
        Calcula o custo interno ao grupo (pickups + drop-offs), sem incluir
        os trechos depot->p0 e pk->depot, usando heurística gulosa.
        """

        metric_fn = getattr(Distances, metric)
        # Inicializa listas mutáveis de origens e destinos
        origins = [p.origin for p in self.passengers]
        destinations = [p.destination for p in self.passengers]

        total_cost = 0.0
        # 1) Escolhe p0: origem mais próxima do depósito
        current_point = min(origins, key=lambda o: metric_fn(*depot, *o))
        origins.remove(current_point)

        # 2) Pick-up interno: visita todos os origins restantes
        while origins:
            next_origin = min(origins, key=lambda o:metric_fn(*current_point, *o))
            total_cost += metric_fn(*current_point, *next_origin)
            current_point = next_origin
            origins.remove(next_origin)

        # 3) Drop-off interno: a partir do último pickup, visita todos os destinations
        while destinations:
            next_dest = min(destinations, key=lambda d: metric_fn(*current_point, *d))
            total_cost += metric_fn(*current_point, *next_dest)
            current_point = next_dest
            destinations.remove(next_dest)

        # 4) Retorna o custo total dos trechos internos
        return total_cost 

class GeoGrouper:
    """Agrupa passageiros com heurística gulosa e restrições de capacidade e diâmetro."""

    def __init__(
        self,
        max_size: int = 5,
        alpha: float = 0.5,
        beta: float = 0.5,
        penalty: float = 10.0,
        delta: float = 1, # distância máxima por grupo, 1 => 100km
        metric: str = "euclidean",
    ) -> None:
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty
        self.delta = delta
        self.metric_fn = getattr(Distances, metric)

    def fit(self, passengers: List[Passenger]) -> List[EAVRPGroup]:
        """Executa o agrupamento e retorna grupos resultantes."""
        # inicia clusters como listas singleton
        clusters: List[List[Passenger]] = [[p] for p in passengers]

        # pré-cálculo de distâncias compostas
        def pair_dist(p_i: Passenger, p_j: Passenger) -> float:
            d_o = self.metric_fn(*p_i.origin, *p_j.origin)
            # print(d_o)
            d_d = self.metric_fn(*p_i.destination, *p_j.destination)
            return self.alpha * d_o + self.beta * d_d
        
        # função de diâmetro de cluster
        def diameter(cluster: List[Passenger]) -> float:
            maxd = 0.0
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    d = pair_dist(cluster[i], cluster[j])
                    if d > maxd:
                        maxd = d
            return maxd

        # função de custo de mesclagem
        def merge_cost(A: List[Passenger], B: List[Passenger]) -> float:
            s = len(A) + len(B)
            # média de distâncias cruzadas
            total = 0.0
            for p in A:
                for q in B:
                    total += pair_dist(p, q)
            avg = total / (len(A) * len(B))
            return (self.max_size - s) * self.penalty + avg

        # loop de fusões
        while True:
            best = None  # (cost, idx_A, idx_B)
            n = len(clusters)
            for i in range(n):
                for j in range(i+1, n):
                    A, B = clusters[i], clusters[j]
                    if len(A) + len(B) > self.max_size:
                        continue
                    # verifica restrição de diâmetro
                    union = A + B
                    if diameter(union) > self.delta:
                        continue
                    c = merge_cost(A, B)
                    if best is None or c < best[0]:
                        best = (c, i, j)
            if best is None:
                break
            _, i, j = best
            # mescla B em A e remove B
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        # cria EAVRPGroup para cada cluster
        return [EAVRPGroup(gid, passenger) for gid, passenger in enumerate(clusters, start=1)]


__all__ = ["GeoGrouper", "EAVRPGroup"]

