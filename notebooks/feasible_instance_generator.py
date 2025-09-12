"""
instance_generator.py (geo‑bounded + radial constraint)
======================================================

Gera dados sintéticos do EA‑VRP dentro de um *bounding box* (default =
região metropolitana de São Paulo) **e**, adicionalmente, limita todos os
pontos (origem, destino, depósito, recarga) a um raio máximo — padrão 2 km —
a partir do depósito principal.  Isso garante que distâncias sejam curtas o
suficiente para uma rota viável com o consumo padrão de bateria.

Parâmetros principais
---------------------
* `bbox`  –  `(lat_min, lat_max, lon_min, lon_max)`  (default SP).
* `max_radius_km` –  raio máximo permitido desde o depósito até qualquer ponto.
* `dest_shift_km` –  deslocamento médio origem→destino dentro desse raio.
* `cluster_radius_km` –  ruído interno de cada cluster (origem/destino).

Exemplo
-------
```python
ig = InstanceGenerator(
        n_passengers=10, n_vehicles=3,
        max_radius_km=2.0,             # <= 2 km do depósito
        dest_shift_km=1.0,             # ~1 km O→D
        seed=42)
vehicles, passengers, depots, rps = ig.build()
```
O código não cria instâncias `Group`; o agrupamento fica a cargo do usuário.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import List, Tuple

import numpy as np

try:
    from utils import Passenger, Vehicle, Depot, RechargePoint
except ImportError as e:  # pragma: no cover
    raise ImportError("instance_generator.py requer utils.py no PYTHONPATH") from e

# ---------- Constantes -------------------------------------------------
MAX_CAPACITY = 5
CONSUMPTION_RATE = 0.10  # % bateria por km
DEFAULT_BBOX = (-24.0, -23.4, -46.8, -46.3)
DEFAULT_MAX_RADIUS_KM = 2.0
DEFAULT_CLUSTER_KM = 0.2   # ruído interno 200 m
DEFAULT_SHIFT_KM = 1.0      # distância O→D ≈1 km

# km ↔ grau (aprox) na latitude −23.6°
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(-23.6))  # ≈102.1 km


def km_to_lat(km: float) -> float:
    return km / KM_PER_DEG_LAT


def km_to_lon(km: float) -> float:
    return km / KM_PER_DEG_LON


class InstanceGenerator:
    """Instância sintética em SP com raio máximo do depósito."""

    def __init__(
        self,
        n_passengers: int,
        n_vehicles: int,
        n_depots: int = 1,
        n_recharges: int = 1,
        bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
        max_radius_km: float = DEFAULT_MAX_RADIUS_KM,
        cluster_radius_km: float = DEFAULT_CLUSTER_KM,
        dest_shift_km: float = DEFAULT_SHIFT_KM,
        seed: int | None = None,
    ) -> None:
        if n_passengers < 1 or n_vehicles < 1 or n_depots < 1:
            raise ValueError("todos n_* devem ser >=1")

        self.n_passengers = n_passengers
        self.n_vehicles = n_vehicles
        self.n_depots = n_depots
        self.n_recharges = n_recharges

        self.lat_min, self.lat_max, self.lon_min, self.lon_max = bbox
        self.max_radius_lat = km_to_lat(max_radius_km)
        self.max_radius_lon = km_to_lon(max_radius_km)
        self.cluster_rad_lat = km_to_lat(cluster_radius_km)
        self.cluster_rad_lon = km_to_lon(cluster_radius_km)
        self.dest_shift_lat = km_to_lat(dest_shift_km)
        self.dest_shift_lon = km_to_lon(dest_shift_km)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def build(self):
        depots = self._make_depots()
        vehicles = self._make_vehicles(depots[0])
        passengers = self._make_passengers(depots[0])
        rps = self._make_rps(depots[0])
        return vehicles, passengers, depots, rps

    # ---------------- Internal builders --------------------------------
    def _make_depots(self):
        lat_c = (self.lat_min + self.lat_max) / 2
        lon_c = (self.lon_min + self.lon_max) / 2
        return [Depot(f"{i}", (lat_c, lon_c)) for i in range(self.n_depots)]

    def _make_vehicles(self, depot: Depot):
        lat, lon = depot.location
        return [Vehicle(f"{i+1}", (lat, lon), battery=10000) for i in range(self.n_vehicles)]

    def _random_partition(self, total):
        parts, rem = [], total
        while rem:
            size = self.rng.randint(1, min(MAX_CAPACITY, rem))
            parts.append(size)
            rem -= size
        self.rng.shuffle(parts)
        return parts

    def _bounded_offset(self, lat_base, lon_base):
        """Aplica ruído mas mantém dentro do raio máximo do depósito."""
        for _ in range(10):  # tenta 10x antes de clipear
            lat = lat_base + self.np_rng.normal(0, self.cluster_rad_lat)
            lon = lon_base + self.np_rng.normal(0, self.cluster_rad_lon)
            if abs(lat - self.lat_depot) <= self.max_radius_lat and abs(lon - self.lon_depot) <= self.max_radius_lon:
                return lat, lon
        # fallback: clipe
        lat = max(min(lat, self.lat_depot + self.max_radius_lat), self.lat_depot - self.max_radius_lat)
        lon = max(min(lon, self.lon_depot + self.max_radius_lon), self.lon_depot - self.max_radius_lon)
        return lat, lon

    def _make_passengers(self, depot: Depot):
        self.lat_depot, self.lon_depot = depot.location
        max_supported = self.n_vehicles * MAX_CAPACITY
        n_pax = min(self.n_passengers, max_supported)
        if n_pax < self.n_passengers:
            warnings.warn(f"Reduzindo passageiros para {n_pax} (capacidade)")

        cluster_sizes = self._random_partition(n_pax)
        passengers = []
        pid = 1
        for size in cluster_sizes:
            # cluster base dentro do raio
            angle = self.rng.uniform(0, 2 * math.pi)
            r_lat = self.max_radius_lat * self.rng.random()
            r_lon = self.max_radius_lon * self.rng.random()
            base_lat = self.lat_depot + r_lat * math.cos(angle)
            base_lon = self.lon_depot + r_lon * math.sin(angle)

            # destino base deslocado dest_shift_km em direção leste
            dest_lat_base = base_lat
            dest_lon_base = base_lon + self.dest_shift_lon
            # garante dentro do raio
            if abs(dest_lon_base - self.lon_depot) > self.max_radius_lon:
                dest_lon_base = self.lon_depot + math.copysign(self.max_radius_lon, dest_lon_base - self.lon_depot)

            for _ in range(size):
                lat_o, lon_o = self._bounded_offset(base_lat, base_lon)
                lat_d, lon_d = self._bounded_offset(dest_lat_base, dest_lon_base)
                passengers.append(Passenger(f"{pid}", (lat_o, lon_o), (lat_d, lon_d)))
                pid += 1
        return passengers

    def _make_rps(self, depot: Depot):
        if self.n_recharges == 0:
            return []
        rps = []
        for i in range(self.n_recharges):
            angle = 2 * math.pi * i / self.n_recharges
            lat = depot.location[0] + self.max_radius_lat * 0.8 * math.cos(angle)
            lon = depot.location[1] + self.max_radius_lon * 0.8 * math.sin(angle)
            rps.append(RechargePoint(f"{i+1}", (lat, lon)))
        return rps


# -------- autoteste rápido --------------------------------------------
if __name__ == "__main__":
    from utils import Group  # type: ignore
    from encoder import QUBOEncoder  # type: ignore
    from classical_solver import ClassicalVRPSolver  # type: ignore

    gen = InstanceGenerator(n_passengers=8, n_vehicles=3, seed=0)
    v, p, d, rp = gen.build()

    # agrupa cada passageiro sozinho (teste viabilidade)
    groups = [Group([pp]) for pp in p]
    enc = QUBOEncoder(v, groups, rp, depot=d[0], penalty_lambda=5.0, penalty_mu=2.0)
    print(ClassicalVRPSolver(enc).best())
