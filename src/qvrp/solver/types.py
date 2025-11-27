# solvers/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


Bit = int
Bitstring = List[Bit]
Energy = float
Counts = Dict[str, int]
Params = Any
SolverMetadata = Dict[str, Any]


@dataclass
class SolverResult:
    """
    Unified result object returned by any quantum solver.

    Attributes
    ----------
    bitstring:
        Selected binary configuration (usually the most probable or lowest-energy one).
    energy:
        Energy value of `bitstring` according to the corresponding QUBO model.
    raw_samples:
        Backend-specific raw result, for example:
            - bitstring counts,
            - parameter trajectories,
            - low-level measurement data.
    metadata:
        Additional information about the run:
            - optimizer history,
            - convergence flags,
            - runtime statistics,
            - instance identifiers, etc.
    """
    bitstring: Bitstring
    energy: Energy
    raw_samples: Any
    metadata: SolverMetadata
