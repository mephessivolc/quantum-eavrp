# geografics.py
from __future__ import annotations
import numpy as np
from typing import Literal

Array = np.ndarray

class Distances:
    """
    Scalar and vector distances.
    Scalar: (x1, y1, x2, y2) -> float
    Vector: broadcastable arrays -> array
    """

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Inputs in degrees. Output in kilometers. Supports scalar or arrays.
        """
        lat1 = np.asarray(lat1, dtype=np.float64)
        lon1 = np.asarray(lon1, dtype=np.float64)
        lat2 = np.asarray(lat2, dtype=np.float64)
        lon2 = np.asarray(lon2, dtype=np.float64)

        R = 6371.0  # km
        phi1 = np.deg2rad(lat1)
        phi2 = np.deg2rad(lat2)
        dphi = np.deg2rad(lat2 - lat1)
        dlmb = np.deg2rad(lon2 - lon1)

        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        return R * c

    @staticmethod
    def euclidean(x1, y1, x2, y2):
        x1 = np.asarray(x1, dtype=np.float64)
        y1 = np.asarray(y1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)
        return np.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def manhattan(x1, y1, x2, y2):
        x1 = np.asarray(x1, dtype=np.float64)
        y1 = np.asarray(y1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)
        return np.abs(x2 - x1) + np.abs(y2 - y1)

    @staticmethod
    def matrix(points: Array, metric: Literal["euclidean","manhattan","haversine"]="euclidean") -> Array:
        """
        NxN matrix of distances. Points shape (N,2).
        For haversine, assume points as (lat, lon) in degrees.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points deve ter shape (N,2)")
        x = pts[:, 0][:, None]  # (N,1)
        y = pts[:, 1][:, None]  # (N,1)
        if metric == "euclidean":
            return Distances.euclidean(x, y, x.T, y.T)
        if metric == "manhattan":
            return Distances.manhattan(x, y, x.T, y.T)
        if metric == "haversine":
            return Distances.haversine(x, y, x.T, y.T)
        raise ValueError("metric inválida")

    @staticmethod
    def pairwise(points_a: Array, points_b: Array, metric: Literal["euclidean","manhattan","haversine"]="euclidean") -> Array:
        """
        Distances between two sets. A shape (NA,2), B shape (NB,2) -> (NA,NB).
        """
        A = np.asarray(points_a, dtype=np.float64)
        B = np.asarray(points_b, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != 2 or B.ndim != 2 or B.shape[1] != 2:
            raise ValueError("points_a e points_b devem ter shape (N,2)")
        x1, y1 = A[:, 0][:, None], A[:, 1][:, None]  # (NA,1)
        x2, y2 = B[:, 0][None, :], B[:, 1][None, :]  # (1,NB)
        if metric == "euclidean":
            return Distances.euclidean(x1, y1, x2, y2)
        if metric == "manhattan":
            return Distances.manhattan(x1, y1, x2, y2)
        if metric == "haversine":
            return Distances.haversine(x1, y1, x2, y2)
        raise ValueError("metric inválida")

if __name__ == "__main__":
    sp = (-23.5505, -46.6333)
    rio = (-22.9068, -43.1729)
    print(f"Haversine: {Distances.haversine(sp[0], sp[1], rio[0], rio[1])}")
    print(f"Euclidean: {Distances.euclidean(0, 0, 3, 4)}")
    print(f"Manhattan: {Distances.manhattan(0, 0, 3, 4)}")
    pts = np.array([[0, 0], [3, 4]])
    print(f"Matrix: {Distances.matrix(pts, metric='euclidean')}")
