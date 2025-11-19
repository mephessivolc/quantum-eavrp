from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from utils import (
    Passenger,
    Group,
    Vehicle,
    Depot,
    RechargePoint,
    Distances,
)

from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)

class GraphBuilder:
    """
    Build a complete graph of passengers, groups, vehicles, depots, and recharge points.
    Edge weights are distances between nodes.
    """

    def __init__(
        self,
        passengers: List[Passenger] | None,
        groups: List[Group] | None,
        depots: List[Depot] | None,
        recharges: List[RechargePoint] | None,
        vehicles: List[Vehicle] | None = None,
        metric: str = "euclidean",
        title_graph: str = "EA-VRP Graph",
        axis: str = "equal"
    ):
        self.passengers = passengers
        self.groups = groups
        self.vehicles = vehicles
        self.depots = depots
        self.recharges = recharges
        self.metric = metric
        self.graph = nx.Graph()

        self.title_graph = title_graph
        self.axis = axis

    # ------------------------------------------------------------------
    def _node_positions_and_colors(self):
        """Return positions and color map for all nodes."""
        # pos = nx.get_node_attributes(self.graph, "pos")
        pos = nx.spring_layout(self.graph, k=0.5, iterations=100)

        color_map = []
        for node in self.graph.nodes:
            if node.startswith("Po"):
                color_map.append("#90EE90")  # light green
            elif node.startswith("Pd"):
                color_map.append("#FFB6C1")  # light red
            elif node.startswith("Go"):
                color_map.append("#008000")  # green
            elif node.startswith("Gd"):
                color_map.append("#FF0000")  # red
            elif node.startswith("R"):
                color_map.append("#FFD700")  # yellow
            elif node.startswith("D"):
                color_map.append("#0000FF")  # blue
            elif node.startswith("Vs"):
                color_map.append("#808080")  # gray
            else:
                color_map.append("#FFFFFF")  # fallback
        return pos, color_map
    
    # ------------------------------------------------------------------
    def _collect_nodes(self):
        """Collect all nodes from passengers, groups, vehicles, depots, and recharge points."""
        nodes = {}

        # Passenger nodes
        if not self.passengers is None:
            for p in self.passengers:
                nodes[p.origin_node.name] = p.origin_node
                nodes[p.destination_node.name] = p.destination_node

        # Group nodes
        if not self.groups is None:
            for g in self.groups:
                nodes[g.origin_node.name] = g.origin_node
                nodes[g.destination_node.name] = g.destination_node

        # Recharge points
        if not self.recharges is None:
            for r in self.recharges:
                nodes[r.node.name] = r.node

        # Depots
        if not self.depots is None:
            for d in self.depots:
                nodes[d.node.name] = d.node

        # Vehicle start nodes
        if not self.vehicles is None:
            for v in self.vehicles:
                nodes[v.start_node.name] = v.start_node

        return nodes


    # ------------------------------------------------------------------
    def build(self) -> nx.Graph:
        """Construct the complete graph with distance-weighted edges."""
        nodes = self._collect_nodes()
        fn = getattr(Distances, self.metric)
        node_items = list(nodes.items())

        for name, node in node_items:
            self.graph.add_node(name, pos=node.loc, type=name[0:2])

        for i in range(len(node_items)):
            n1, node1 = node_items[i]
            for j in range(i + 1, len(node_items)):
                n2, node2 = node_items[j]
                dist = float(fn(*node1.loc, *node2.loc))
                self.graph.add_edge(n1, n2, weight=dist)
        return self.graph

    # ------------------------------------------------------------------
    def draw(self, figsize=(8, 6)) -> None:
        """Display the graph interactively."""
        pos, color_map = self._node_positions_and_colors()
        plt.figure(figsize=figsize)
        nx.draw(
            self.graph,
            pos,
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=8,
        )
        labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels={k: f"{v:.1f}" for k, v in labels.items()}, font_size=6
        )
        plt.title(self.title_graph)
        plt.axis(self.axis)
        plt.show()

    # ------------------------------------------------------------------
    def save(
            self, 
            filename: str, 
            folder: str | Path = "outputs", 
            figsize=(8, 6),
            show_weights: bool = False,
        ) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        if not filename.lower().endswith(".png"):
            filename += ".png"
        file_path = folder_path / filename

        pos, color_map = self._node_positions_and_colors()
        plt.figure(figsize=figsize)
        nx.draw(
            self.graph,
            pos,
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=8,
        )

        # # --- safe edge labels ---
        if show_weights:
            labels = nx.get_edge_attributes(self.graph, "weight")
            for (u, v), w in labels.items():
                (x1, y1), (x2, y2) = pos[u], pos[v]
                plt.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    f"{w:.1f}",
                    fontsize=6,
                    color="black",
                    ha="center",
                    va="center",
                )

        plt.title(self.title_graph)
        plt.axis(self.axis)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graph image saved at: {file_path.resolve()}")


# ---------------------------------------------------------------------
# Quick test when executed directly
if __name__ == "__main__":
    # Create 3 passengers, 3 groups, 3 vehicles, 3 depots, and 3 recharge points
    passengers = [Passenger(i + 1, (i, 0.0), (i + 0.5, 1.0)) for i in range(3)]
    groups = [Group(i + 1, [passengers[i]]) for i in range(3)]
    vehicles = [Vehicle(i + 1, (0.0, i * 1.0), 100.0, 1.5, 20.0) for i in range(3)]
    depots = [Depot(i + 1, (-1.0, i * 2.0)) for i in range(3)]
    recharges = [RechargePoint(i + 1, (2.0, i * 1.5)) for i in range(3)]

    builder = GraphBuilder(passengers, groups, depots, recharges)
    G = builder.build()
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    builder.save("test_create.png")