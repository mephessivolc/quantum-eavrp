"""classical_solver.py — versão com *fallback* de custo

- Mantém a interface `ClassicalVRPSolver(encoder)` usada no projeto.
- Usa `encoder.encode()` → (Q, offset, meta) e guarda `self.offset`.
- `best()` e `k_best()` agora preenchem `cost` mesmo quando `encoder.cost(bits)`
  não existe (ou retorna NaN), **desde que a solução seja factível**:
  `cost = energy - offset` (energia do QUBO sem penalidade constante).

Observação:
- Se você implementar `encoder.cost(bits)` (recomendado), ele será usado
  preferencialmente; o *fallback* só entra quando `cost` vier NaN/None.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import dimod


def _safe_cost(encoder: Any, bits: str) -> float:
    cost_fn = getattr(encoder, "cost", None)
    if callable(cost_fn):
        try:
            val = cost_fn(bits)
            return float(val) if val is not None else float("nan")
        except Exception:
            return float("nan")
    return float("nan")


def _sample_to_bitstring(sample: Dict[Any, int], var_order: List[Any]) -> str:
    # Converte {var:0/1} → "0101..." seguindo a ordem de variáveis fornecida
    return "".join("1" if int(sample.get(v, 0)) else "0" for v in var_order)


class ClassicalVRPSolver:
    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.bqm: Optional[dimod.BQM] = None
        self.response: Optional[dimod.SampleSet] = None
        self.variables: Optional[List[Any]] = None
        self.offset: float = 0.0

    def _build_bqm(self) -> None:
        # Espera-se: encode() → (Q, offset, meta)
        Q, offset, _meta = self.encoder.encode()
        self.offset = float(offset) if offset is not None else 0.0
        self.bqm = dimod.BQM.from_qubo(Q, offset=self.offset)
        self.variables = list(self.bqm.variables)

    def solve(self) -> None:
        if self.bqm is None:
            self._build_bqm()
        assert self.bqm is not None
        n = len(self.bqm.variables)
        # Exato para instâncias pequenas; caso contrário, SA como *fallback*
        if n <= 25:
            sampler = dimod.ExactSolver()
            self.response = sampler.sample(self.bqm)
        else:
            sampler = dimod.SimulatedAnnealingSampler()
            self.response = sampler.sample(self.bqm, num_reads=5000)

    def best(self) -> Dict[str, Any]:
        if self.response is None:
            raise RuntimeError("Chame solve() antes de best().")
        rec = self.response.first
        var_order = self.variables or list(rec.sample.keys())
        bits = _sample_to_bitstring(rec.sample, var_order)

        # Feasibility pelo encoder
        feas = False
        try:
            feas = bool(self.encoder.is_feasible(bits))
        except Exception:
            feas = False

        # Custo preferencial via encoder.cost(bits)
        cost = _safe_cost(self.encoder, bits)
        # Fallback: se viável e custo ausente/NaN, usa energia- offset
        if (not np.isfinite(cost)) and feas:
            cost = float(rec.energy) - float(self.offset)

        # Rotas decodificadas (se o encoder expõe interpret())
        routes = None
        try:
            interpret = getattr(self.encoder, "interpret", None)
            routes = interpret(bits) if callable(interpret) else None
        except Exception:
            routes = None

        return {
            "bits": bits,
            "energy": float(rec.energy),
            "feasible": feas,
            "cost": float(cost) if np.isfinite(cost) else float("nan"),
            "routes": routes,
        }

    def k_best(self, k: int = 5) -> List[Dict[str, Any]]:
        if self.response is None:
            raise RuntimeError("Chame solve() antes de k_best().")
        out: List[Dict[str, Any]] = []
        size = min(k, len(self.response))
        var_order = self.variables or list(self.response.variables)
        for i in range(size):
            rec = self.response.record[i]
            sample = {v: rec.sample[j] for j, v in enumerate(self.response.variables)}
            bits = _sample_to_bitstring(sample, var_order)
            try:
                feas = bool(self.encoder.is_feasible(bits))
            except Exception:
                feas = False
            cost = _safe_cost(self.encoder, bits)
            if (not np.isfinite(cost)) and feas:
                cost = float(rec.energy) - float(self.offset)
            out.append({
                "bits": bits,
                "energy": float(rec.energy),
                "feasible": feas,
                "cost": float(cost) if np.isfinite(cost) else float("nan"),
            })
        return out
