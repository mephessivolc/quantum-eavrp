from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from geographics import Distances

Coord = Tuple[float, float]
DistanceFn = Callable[[float, float, float, float], float]

# ---------- Basics ----------
@dataclass(frozen=True)
class Node:
    name: str
    loc: Coord
    def as_tuple(self) -> Tuple[str, Coord]: 
        return self.name, self.loc

@dataclass(frozen=True)
class Arc:
    u: Node
    v: Node
    cost: float
    def as_tuple(self) -> Tuple[str, str, float]: 
        return self.u.name, self.v.name, self.cost

# ---------- Mixins ----------
def _get_metric_fn(metric: str) -> DistanceFn:
    # Search for the function in Geographics.Distances by name
    fn = getattr(Distances, metric, None)
    if fn is None or not callable(fn):
        raise ValueError(f"métrica inválida: {metric}")
    return fn  # type: ignore[return-value]

class Locatable:
    _node: Node
    @property
    def location(self) -> Coord: 
        return self._node.loc
    
    def distance_to(self, point: Coord, metric: str = "euclidean") -> float:
        fn = _get_metric_fn(metric)
        x1, y1 = self._node.loc; x2, y2 = point
        return float(fn(x1, y1, x2, y2))
    
    def get_nodes(self) -> List[Node]: 
        return [self._node]

class OriginDest:
    _o: Node
    _d: Node
    def get_nodes(self) -> List[Node]: 
        return [self._o, self._d]
    def get_internal_arc(self, metric: str = "euclidean") -> Arc:
        fn = _get_metric_fn(metric)
        x1, y1 = self._o.loc; x2, y2 = self._d.loc
        return Arc(self._o, self._d, float(fn(x1, y1, x2, y2)))
    
    @property
    def origin_node(self) -> Node: 
        return self._o
    
    @property
    def destination_node(self) -> Node: 
        return self._d
    
    def get_node_ids(self) -> List[str]: 
        return [self._o.name, self._d.name]

# ---------- Domínio ----------
class Passenger(OriginDest):
    def __init__(self, pid: int, origin: Coord, destination: Coord):
        self._id = pid
        self._o = Node(f"Po{pid}", origin)
        self._d = Node(f"Pd{pid}", destination)
    @property
    def id(self) -> int: return self._id

class Group(OriginDest):
    def __init__(self, gid: int, passengers: List[Passenger]):
        if not passengers: raise ValueError("Group requer >= 1 passageiro")
        self._id = gid
        self._passengers = passengers
        # Origem/destino do grupo herdados do primeiro passageiro (regra atual)
        self._o = Node(f"Go{gid}", passengers[0].origin_node.loc)
        self._d = Node(f"Gd{gid}", passengers[0].destination_node.loc)

    def __iter__(self) -> Iterable[Passenger]: 
        return iter(self._passengers)

    @property
    def id(self) -> int: 
        return self._id
    
    @property
    def passengers(self) -> List[Passenger]: 
        return self._passengers
    
    def get_internal_arcs(self, metric: str = "euclidean") -> List[Arc]:
        return [p.get_internal_arc(metric) for p in self._passengers]

class RechargePoint(Locatable):
    def __init__(self, rid: int, location: Coord):
        self._id = rid
        self._node = Node(f"R{rid}", location)
    
    @property
    def id(self) -> int: 
        return self._id
    
    @property
    def node(self) -> Node: 
        return self._node

class Depot(Locatable):
    def __init__(self, did: int, location: Coord):
        self._id = did
        self._node = Node(f"D{did}", location)

    @property
    def id(self) -> int: 
        return self._id
    
    @property
    def node(self) -> Node: 
        return self._node

class Vehicle(Locatable):
    def __init__(
        self,
        vid: int,
        start_location: Coord,
        battery: float,
        consumption_per_km: float,
        min_charge: float,
    ):
        self._id = vid
        self._node = Node(f"Vs{vid}", start_location)
        self._battery = float(battery)
        self._cons = float(consumption_per_km)
        self._min = float(min_charge)

    @property
    def id(self) -> int: 
        return self._id
    
    @property
    def start_node(self) -> Node: 
        return self._node
    
    @property
    def battery(self) -> float: 
        return self._battery
    
    @property
    def consumption_per_km(self) -> float: 
        return self._cons
    
    @property
    def min_charge(self) -> float: 
        return self._min
    
    def energy_needed(self, dist_km: float) -> float: 
        return dist_km * self._cons
    
    def needs_recharge(self, dist_km: float) -> bool:
        return self._battery - self.energy_needed(dist_km) < self._min

# ---------- Utilidades orientadas a objetos ----------
def make_internal_graph(passengers: List[Passenger], metric: str = "euclidean") -> Dict[str, List[Arc]]:
    return {str(p.id): [p.get_internal_arc(metric)] for p in passengers}

def pairwise_arcs(nodes: List[Node], metric: str = "euclidean") -> List[Arc]:
    fn = _get_metric_fn(metric)
    arcs: List[Arc] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            x1, y1 = u.loc; x2, y2 = v.loc
            arcs.append(Arc(u, v, float(fn(x1, y1, x2, y2))))
    return arcs


if __name__ == "__main__":
    # Smoke test: create and display three instances of each class

    print("=== Nodes ===")
    nodes = [Node(f"N{i+1}", (float(i), float(i) + 0.5)) for i in range(3)]
    for n in nodes:
        print("Node:", n.as_tuple())

    print("\n=== Arcs ===")
    arcs = [
        Arc(
            nodes[i],
            nodes[(i + 1) % 3],
            float(Distances.euclidean(*nodes[i].loc, *nodes[(i + 1) % 3].loc)),
        )
        for i in range(3)
    ]
    for a in arcs:
        print("Arc:", a.as_tuple())

    print("\n=== Passengers ===")
    passengers = [Passenger(i + 1, (i * 1.0, 0.0), (i * 1.0 + 0.5, 1.0)) for i in range(3)]
    for p in passengers:
        arc = p.get_internal_arc("euclidean")
        print(f"Passenger#{p.id} nodes:", p.get_node_ids(), "internal_arc:", arc.as_tuple())

    print("\n=== Groups ===")
    groups = [Group(i + 1, [passengers[i]]) for i in range(3)]
    for g in groups:
        node_names = [n.name for n in g.get_nodes()]
        internal_arcs = [a.as_tuple() for a in g.get_internal_arcs("euclidean")]
        print(f"Group#{g.id} nodes:", node_names, "internal_arcs:", internal_arcs)

    print("\n=== RechargePoints ===")
    rps = [RechargePoint(i + 1, (i * 2.0, -i * 1.0)) for i in range(3)]
    for r in rps:
        print(f"RechargePoint#{r.id} node:", r.node.as_tuple())

    print("\n=== Depots ===")
    depots = [Depot(i + 1, (-i * 1.0, i * 2.0)) for i in range(3)]
    for d in depots:
        print(f"Depot#{d.id} node:", d.node.as_tuple())

    print("\n=== Vehicles ===")
    vehicles = [
        Vehicle(
            vid=i + 1,
            start_location=(i * 1.0, i * 1.0),
            battery=100.0 - 10.0 * i,
            consumption_per_km=1.5,
            min_charge=20.0,
        )
        for i in range(3)
    ]
    for v in vehicles:
        target = (v.start_node.loc[0] + 1.0, v.start_node.loc[1] + 1.0)
        dist = v.distance_to(target, metric="euclidean")
        print(
            f"Vehicle#{v.id} start={v.start_node.as_tuple()} battery={v.battery:.1f}% "
            f"dist_to_target={dist:.2f}km needs_recharge={v.needs_recharge(dist)}"
        )
