# instance_generator.py
from __future__ import annotations
import random
import math
from typing import List, Tuple
from qvrp.domain.model import Passenger, Vehicle, RechargePoint, Depot


class InstanceGenerator:
    """
    Random instance generator for EA-VRP using realistic geographic coordinates.
    Points are sampled within a latitude/longitude bounding box and distances are realistic (km).
    """

    def __init__(
        self,
        n_passengers: int,
        n_vehicles: int,
        n_recharges: int,
        n_depots: int,
        area_lat: Tuple[float, float] = (-23.7, -23.4),
        area_lon: Tuple[float, float] = (-46.8, -46.4),
        seed: int | None = None,
    ):
        """
        Args:
            n_passengers: number of passengers
            n_vehicles: number of vehicles
            n_recharges: number of recharge points
            n_depots: number of depots
            area_lat: latitude interval (e.g., (-23.7, -23.4))
            area_lon: longitude interval (e.g., (-46.8, -46.4))
            seed: optional random seed
        """
        self.n_passengers = n_passengers
        self.n_vehicles = n_vehicles
        self.n_recharges = n_recharges
        self.n_depots = n_depots
        self.area_lat = area_lat
        self.area_lon = area_lon
        self.seed = seed

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    def _rand_coord(self) -> Tuple[float, float]:
        """Generate a random (lat, lon) coordinate inside defined bounding box."""
        lat = random.uniform(*self.area_lat)
        lon = random.uniform(*self.area_lon)
        return (lat, lon)

    # ------------------------------------------------------------------
    def _offset_coord(self, base: Tuple[float, float], max_km: float = 5.0) -> Tuple[float, float]:
        """
        Generate a coordinate near 'base' within a distance of up to max_km.

        Uses simple equirectangular approximation:
        1 degree latitude ≈ 111 km
        1 degree longitude ≈ 111 * cos(lat) km
        """
        lat, lon = base
        delta_lat = (random.uniform(-max_km, max_km)) / 111.0
        delta_lon = (random.uniform(-max_km, max_km)) / (111.0 * abs(math.cos(math.radians(lat))) + 1e-6) # 1e-6 avoid ZeroDivisionError
        new_lat = max(min(lat + delta_lat, self.area_lat[1]), self.area_lat[0])
        new_lon = max(min(lon + delta_lon, self.area_lon[1]), self.area_lon[0])
        return (new_lat, new_lon)

    # ------------------------------------------------------------------
    def create(self):
        """
        Create random EA-VRP components with realistic geo-coordinates.

        Returns:
            passengers (List[Passenger]),
            vehicles (List[Vehicle]),
            recharges (List[RechargePoint]),
            depots (List[Depot])
        """
        import math

        # --- passengers ---
        passengers: List[Passenger] = []
        for i in range(self.n_passengers):
            origin = self._rand_coord()
            destination = self._offset_coord(origin, max_km=random.uniform(1, 8))
            passengers.append(Passenger(i + 1, origin, destination))

        # --- vehicles ---
        vehicles: List[Vehicle] = []
        for i in range(self.n_vehicles):
            start_location = self._rand_coord()
            battery = random.uniform(80.0, 100.0)
            consumption = random.uniform(0.2, 0.4)
            min_charge = 20.0
            vehicles.append(Vehicle(i + 1, start_location, battery, consumption, min_charge))

        # --- recharge points ---
        recharges: List[RechargePoint] = []
        for i in range(self.n_recharges):
            location = self._rand_coord()
            recharges.append(RechargePoint(i + 1, location))

        # --- depots ---
        depots: List[Depot] = []
        for i in range(self.n_depots):
            location = self._rand_coord()
            depots.append(Depot(i + 1, location))

        return passengers, vehicles, recharges, depots


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    import math
    from graphs import GraphBuilder

    # Example: simulate São Paulo urban area bounding box
    generator = InstanceGenerator(
        n_passengers=5,
        n_vehicles=3,
        n_recharges=2,
        n_depots=1,
        area_lat=(-23.70, -23.40),
        area_lon=(-46.80, -46.40),
        seed=42,
    )

    passengers, vehicles, recharges, depots = generator.create()

    print(f"Generated {len(passengers)} passengers")
    for p in passengers:
        print(f"Passenger {p.id} origin={p.origin_node} destination={p.destination_node}")

    print(f"Generated {len(vehicles)} vehicles")
    for v in vehicles:
        print(f"Vehicle {v.id} start={v.start_node}")

    print(f"Generated {len(recharges)} recharge points")
    for r in recharges:
        print(f"Recharge Points {r.id} location={r.node}")
    
    print(f"Generated {len(depots)} depots")
    for d in depots:
        print(f"Depots {d.id} location={d.node}")


    builder = GraphBuilder(passengers, None, depots, recharges)
    G = builder.build()
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    builder.save("test_create_instance_generator.png")
    