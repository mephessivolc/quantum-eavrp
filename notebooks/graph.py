from __future__ import annotations

"""graph_builder.py
====================
Cria e salva um grafo EA-VRP usando **NetworkX**.

Elementos representados
----------------------
* **Origem passageiro**   - rótulo ``O{id}``  cor *lightblue*
* **Destino passageiro**  - rótulo ``D{id}``  cor *lightgreen*
* **Origem do grupo**     - rótulo ``GO{id}`` cor *blue*
* **Destino do grupo**    - rótulo ``GD{id}`` cor *green*
* **Ponto de recarga**    - rótulo ``R{id}``  cor *red*

Arestas
~~~~~~~
Cada par origem→destino recebe uma aresta com rótulo igual à distância
Euclidiana (ou Haversine, se preferir) obtida em
``geografics.Distances``.

Função principal
~~~~~~~~~~~~~~~~
``build_eavrp_graph(passengers, groups, stations, out_png, metric='euclidean')``

* **passengers** - lista de ``Passenger``
* **groups** - lista de ``EAVRPGroup``
* **stations** - lista de tuplas ``(lat, lon)`` **ou** objetos com atributo
  coordenada (``coord``, ``location``, ``pos`` ou similar)
* **out_png** - caminho do arquivo de saída
* **metric** - 'euclidean' ou outro método de ``Distances``
"""

"""graph_builder.py
====================
Cria e salva um grafo EA-VRP completo com **NetworkX**.
"""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt # type: ignore
import networkx as nx # type: ignore

from geografics import Distances
from grouping import EAVRPGroup
from utils import Passenger, Depot, RechargePoint

Coord = Tuple[float, float]
Station = Union[Coord, object]


def calc_distance(fn, o: Tuple[float,float], d: Tuple[float,float], dim=100) -> float:
    return fn(*o, *d) * dim

class Graph:
    def __init__(
        self,
        passengers: List[Passenger],
        groups: List[EAVRPGroup],
        recharge_points: List[RechargePoint],
        depots: List[Depot],
        metric: str = "euclidean"
    ) -> None:

        self.metric_fn = getattr(Distances, metric)
        self.passengers = passengers
        self.groups = groups
        self.recharge_points = recharge_points
        self.depots = depots

        self.nodes = []
        self.arcs = []  # [(i, j, cost)]

        self.G = nx.Graph()
        self._build_graph()
        self._build_complete_arcs()

    def _build_graph(self):
        # Adiciona nós de passageiros
        for p in self.passengers:
            o_node = p.origin_node
            d_node = p.destination_node
            self.G.add_node(o_node, pos=tuple(p.origin), color="lightblue", label=o_node)
            self.G.add_node(d_node, pos=tuple(p.destination), color="lightgreen", label=d_node)
            self.nodes.extend([o_node, d_node])

        # Adiciona grupos
        for g in self.groups:
            go_node = g.origin_node
            gd_node = g.destination_node
            self.G.add_node(go_node, pos=tuple(g.origin), color="blue", label=go_node)
            self.G.add_node(gd_node, pos=tuple(g.destination), color="green", label=gd_node)
            self.nodes.extend([go_node, gd_node])

        # Adiciona estações de recarga
        for r in self.recharge_points:
            r_node = r.node
            self.G.add_node(r_node, pos=tuple(r.location), color="red", label=r_node)
            self.nodes.append(r_node)

        # Adiciona depósitos
        for d in self.depots:
            d_node = d.node
            self.G.add_node(d_node, pos=tuple(d.location), color='yellow', label=d_node)
            self.nodes.append(d_node)

    def _build_complete_arcs(self):
        added = set()
        for i in self.nodes:
            for j in self.nodes:
                if i == j or not isinstance(i, str) or not isinstance(j, str):
                    continue
                key = tuple(sorted([i, j]))

                if key in added:
                    continue
                pos_i = self.G.nodes[i]['pos']
                pos_j = self.G.nodes[j]['pos']
                cost = calc_distance(self.metric_fn, pos_i, pos_j)
                self.arcs.append((key[0], key[1], cost))
                self.G.add_edge(key[0], key[1], weight=cost)
                added.add(key)
    
    def get_nodes(self):
        return self.nodes
    
    def get_arcs(self):
        return self.arcs

    def _create_graph(self):
        pos = nx.get_node_attributes(self.G, "pos")
        colors = [data["color"] for _, data in self.G.nodes(data=True)]
        labels = nx.get_node_attributes(self.G, "label")
        edge_labels = {
            e: f"{d['weight']:.1f}"
            for e, d in self.G.edges.items()
            if "weight" in d
        }

        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, node_color=colors, with_labels=True, labels=labels,
                node_size=500, font_size=8, arrows=True)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7)

        plt.gca().text(
            0.02, 0.02,
            "Escala 1:100",
            transform=plt.gca().transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
        )

        plt.axis("off")
        return plt

    def save(self, filename: str | None):
        plt = self._create_graph()
        if not filename:
            filename = "default_name_graph.png"
        plt.savefig(filename, dpi=300)

    def show(self):
        plt = self._create_graph()
        plt.show()
        plt.close()

__all__ = ["Graph"]

if __name__ == "__main__":
    from feasible_instance_generator import InstanceGenerator
    from grouping import GeoGrouper
    from pathlib import Path

    OUTDIR = Path("outputs")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    filename_hist = OUTDIR / "test_generate_graph.png"

    diameter = 0.004
   
    vehicles, passengers, depots, recharge_points = InstanceGenerator(
        n_passengers=10,
        n_vehicles=8,
        n_depots=1,
        n_recharges=5,
    ).build()

    groups = GeoGrouper(delta=diameter).fit(passengers)

    try: 
        graph = Graph(
            passengers=passengers, 
            groups=groups,
            recharge_points=recharge_points,
            depots=depots)

        print(f"Contrução da classe concluída com sucesso: {graph}")
    
    except Exception as e:
        raise Exception(f"Erro de construção de classe: {e}")


    try:
        print(f"Grafo criado corretamente: {graph.get_arcs()}")
    
    except Exception as e:
        raise Exception(f"Erro de impressão do grafo: {e}")
    
    try:
        graph.save(filename_hist)
        print(f"Imagem salva com sucesso")

    except Exception as e:
        raise Exception(f"Erro de salvamento de imagem: {e}")