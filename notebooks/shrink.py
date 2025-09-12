from typing import List, Dict, Tuple, Set
from utils import Passenger, Group
from geografics import Distances
import numpy as np


class Shrink:
    """
    Implementação do estágio de coarsening do algoritmo Shrink (Sadri et al., 2017),
    adaptado para o modelo EA-CVRP com Passenger → Group (supernós).

    Preserva distâncias aproximadas ao fundir pares de nós com base no erro esperado nulo.
    """

    def __init__(self, 
                 passengers: List[Passenger], 
                 groups: Dict[int, Group],
                 graph: Dict[int, Dict[int, float]],
                 metric: str = "euclidean", 
                 compression_ratio: float = 0.5):
        self.passengers = passengers
        self.metric_fn = getattr(Distances, metric)
        self.compression_ratio = max(0.01, min(1.0, compression_ratio))
        self.groups = groups
        self.graph = graph

    def _build_initial_graph(self) -> Dict[int, Dict[int, float]]:
        """
        Cria um grafo completo não direcionado onde cada nó é um passenger.id.
        As arestas têm peso baseado na média entre distância origem-origem e destino-destino.
        """
        G = {}
        for i, pi in self.passengers:
            G[pi.id] = {}
            for j, pj in self.passengers:
                if i == j:
                    continue
                do = self.metric_fn(*pi.origin, *pj.origin)
                dd = self.metric_fn(*pi.destination, *pj.destination)
                G[pi.id][pj.id] = 0.5 * (do + dd)
        return G

    def run(self) -> List[Group]:
        """
        Executa o coarsening até atingir o número desejado de supernós.
        Cada fusão recalcula pesos preservando o valor médio do caminho local.
        """
        target = max(1, int(self.compression_ratio * len(self.groups)))

        while len(self.groups) > target:
            best_pair = self._select_best_pair()
            if best_pair is None:
                break
            i, j = best_pair
            self._merge_nodes(i, j)

        return list(self.groups.values())

    def _select_best_pair(self) -> Tuple[int, int] | None:
        """
        Seleciona o par (i, j) de nós adjacentes com menor erro estimado local.
        Aqui usamos a heurística de menor grau × peso.
        """
        min_score = np.inf
        best = None
        for i in self.graph:
            for j in self.graph[i]:
                deg_i = len(self.graph[i])
                deg_j = len(self.graph[j])
                w = self.graph[i][j]
                score = (deg_i + deg_j) * w
                if score < min_score:
                    min_score = score
                    best = (i, j)
        return best

    def _merge_nodes(self, i: int, j: int):
        """
        Mescla os nós i e j em um novo supernó, atualiza o grafo e os grupos.
        Novo peso com vizinhos é a média simples das contribuições via i e j.
        """
        new_id = min(i, j)  # ID determinístico
        old_ids = {i, j}
        new_members = self.groups[i].passengers + self.groups[j].passengers
        self.groups[new_id] = Group(new_id, new_members)

        # atualiza grafo
        neighbors = (set(self.graph[i]) | set(self.graph[j])) - old_ids
        self.graph[new_id] = {}
        for k in neighbors:
            wi = self.graph[i].get(k, np.inf)
            wj = self.graph[j].get(k, np.inf)
            new_w = min(wi, wj)
            self.graph[new_id][k] = new_w
            self.graph[k][new_id] = new_w

        # remove antigos
        for x in old_ids:
            self.graph.pop(x, None)
            for n in self.graph:
                self.graph[n].pop(x, None)
            self.groups.pop(x, None)

