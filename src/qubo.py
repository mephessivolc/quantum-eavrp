from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple, Optional
import numpy as np


Index = int
NodeKey = Tuple[Any, Any, Any]  # e.g. (vehicle_id, origin_node, destination_node)


@dataclass
class QUBOBase:
    """
    Base class for QUBO models of the form:

        E(x) = x^T Q x + c^T x + offset

    This class is backend-agnostic and focuses on:
    - storing Q, c, offset, and variable indexing,
    - providing core, universal methods:
        * construction from encoder output
        * copying
        * classical energy evaluation
        * human-readable representation
        * QUBO -> Ising transformation

    Backend-specific subclasses (e.g. QUBOPennyLane, QUBOBraket)
    will implement the conversion to concrete Hamiltonian objects.
    """

    Q: np.ndarray
    c: np.ndarray
    offset: float
    var_index: Dict[NodeKey, Index]

    # ==========================
    # BASIC PROPERTIES
    # ==========================

    @property
    def num_vars(self) -> int:
        """
        Return the number of binary variables in the QUBO.

        This is assumed to be consistent with:
        - Q.shape == (n, n)
        - c.shape == (n,)
        """
        return int(self.Q.shape[0])
    # ==========================
    # CONSTRUCTION HELPERS
    # ==========================

    @classmethod
    def from_encoder_output(
        cls,
        Q: np.ndarray,
        c: np.ndarray,
        offset: float,
        var_index: Dict[NodeKey, Index],
    ) -> "QUBOBase":
        """
        Build a QUBOBase (or subclass) instance from the raw output of an Encoder.

        Typical usage:
        --------------
            Q, c, offset, var_index = encoder.encode()
            qubo = QUBOBase.from_encoder_output(Q, c, offset, var_index)

        Responsibilities:
        -----------------
        - enforce basic shape consistency between Q and c,
        - ensure Q is square,
        - optionally warn about strong asymmetries in Q (later, if desired),
        - store copies of Q, c and var_index to protect against external
          mutations.
        """
        # 1. Convert to numpy arrays (if they are not already)
        Q_arr = np.array(Q, dtype=float, copy=True)
        c_arr = np.array(c, dtype=float, copy=True)

        # 2. Basic shape checks
        if Q_arr.ndim != 2:
            raise ValueError(f"Q must be a 2D array, got ndim={Q_arr.ndim}.")

        n_rows, n_cols = Q_arr.shape
        if n_rows != n_cols:
            raise ValueError(
                f"Q must be a square matrix, got shape={Q_arr.shape}."
            )

        if c_arr.ndim != 1:
            raise ValueError(f"c must be a 1D array, got ndim={c_arr.ndim}.")

        if c_arr.shape[0] != n_rows:
            raise ValueError(
                f"Inconsistent dimensions between Q and c: "
                f"Q is {Q_arr.shape}, c has length {c_arr.shape[0]}."
            )

        # 3. Copy var_index defensivamente (dict raso é suficiente aqui)
        var_index_copy: Dict[NodeKey, Index] = dict(var_index)

        return cls(
            Q=Q_arr,
            c=c_arr,
            offset=float(offset),
            var_index=var_index_copy,
        )

    def copy(self) -> "QUBOBase":
        """
        Return a deep logical copy of this QUBO object.

        - Q and c are copied as independent numpy arrays.
        - var_index is shallow-copied (sufficient because keys/values are immutable).
        - offset is copied as a float.
        """
        Q_copy = np.array(self.Q, dtype=float, copy=True)
        c_copy = np.array(self.c, dtype=float, copy=True)
        var_index_copy = dict(self.var_index)

        return QUBOBase(
            Q=Q_copy,
            c=c_copy,
            offset=float(self.offset),
            var_index=var_index_copy,
        )


    def energy(self, bitstring: Iterable[int]) -> float:
        """
        Evaluate the QUBO energy:

            E(x) = x^T Q x + c^T x + offset

        Parameters
        ----------
        bitstring:
            Iterable of 0/1 integers (or bools). Length must be equal to num_vars.

        Returns
        -------
        float:
            The scalar energy value for this assignment.

        Notes
        -----
        - This method is purely classical.
        - Extremely useful for debugging and verifying constraint correctness
          in small instances.
        """
        x = np.array(list(bitstring), dtype=float)

        if x.shape[0] != self.num_vars:
            raise ValueError(
                f"Bitstring length {x.shape[0]} does not match QUBO size {self.num_vars}."
            )

        # Compute x^T Q x (quadratic term)
        quad = x @ self.Q @ x

        # Compute c^T x (linear term)
        linear = self.c @ x

        return float(quad + linear + self.offset)

    # ==========================
    # HUMAN-READABLE VIEW
    # ==========================

    def human_readable(self, max_terms: Optional[int] = None) -> str:
        """
        Build a human-readable representation of the QUBO energy function.

        Output format (example):
        ------------------------
        E(x) = offset
             + 3.2 * x[v0, 1, 2]
             - 1.1 * x[v1, 5, 6]
             + 0.8 * x[v0, 2, 3] x[v0, 3, 4]
             ...

        Parameters
        ----------
        max_terms:
            If not None, limit total number of printed terms. When the limit
            is reached, output stops with an ellipsis.

        Notes
        -----
        - Only terms with non-zero coefficients are shown.
        - Variable names use var_index to map back to semantic names.
        """
        lines = []
        lines.append("E(x) =")

        # Add offset term
        if self.offset != 0:
            lines.append(f"    {self.offset:+.6f}")

        count = 0  # number of printed terms
        n = self.num_vars

        # Invert var_index to map idx -> semantic key
        rev_index = {idx: key for key, idx in self.var_index.items()}

        # ----------------------------------------
        # Linear terms: c[i] * x_i
        # ----------------------------------------
        for i in range(n):
            coef = self.c[i]
            if abs(coef) < 1e-12:  # ignore zero or tiny noise
                continue

            # get semantic name if available
            if i in rev_index:
                v, a, b = rev_index[i]
                name = f"x[v{v}, {a}, {b}]"
            else:
                name = f"x[{i}]"

            lines.append(f"    {coef:+.6f} * {name}")
            count += 1

            if max_terms is not None and count >= max_terms:
                lines.append("    ... (truncated)")
                return "\n".join(lines)

        # ----------------------------------------
        # Quadratic terms: Q[i,j] * x_i x_j
        # ----------------------------------------
        for i in range(n):
            for j in range(i + 1, n):
                coef = self.Q[i, j]
                if abs(coef) < 1e-12:
                    continue

                # semantic names
                if i in rev_index:
                    v1, a1, b1 = rev_index[i]
                    n1 = f"x[v{v1}, {a1}, {b1}]"
                else:
                    n1 = f"x[{i}]"

                if j in rev_index:
                    v2, a2, b2 = rev_index[j]
                    n2 = f"x[v{v2}, {a2}, {b2}]"
                else:
                    n2 = f"x[{j}]"

                lines.append(f"    {coef:+.6f} * {n1} {n2}")
                count += 1

                if max_terms is not None and count >= max_terms:
                    lines.append("    ... (truncated)")
                    return "\n".join(lines)

        return "\n".join(lines)

    # ==========================
    # QUBO → ISING TRANSFORMATION
    # ==========================

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert the QUBO (Q, c, offset) into an Ising model (h, J, const).

        Mapping
        -------
        We use the standard binary → spin transformation:

            x_i ∈ {0, 1},  z_i ∈ {+1, -1}
            x_i = (1 - z_i) / 2

        The QUBO energy is:

            E_Q(x) = x^T Q x + c^T x + offset

        We seek an equivalent Ising form:

            E_I(z) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j

        such that E_Q(x(z)) == E_I(z) for all z ∈ {+1, -1}^n.

        Returns
        -------
        h:
            1D array of shape (n,) with local field coefficients h_i.
        J:
            2D array of shape (n, n) with pairwise couplings J_ij.
            By convention, J_ii = 0 and J_ij = J_ji, and only i<j
            entries are physically relevant.
        const:
            Constant energy shift such that the two energies match.

        Notes
        -----
        - This method assumes Q is symmetric. If it is not perfectly
          symmetric due to numerical noise, the user should symmetrize
          Q beforehand (e.g., in the encoder finalization).
        - The returned (h, J, const) are backend-agnostic and can be
          used by different quantum frameworks to build Hamiltonians.
        """
        n = self.num_vars

        # Ensure numpy arrays with float dtype
        Q = np.array(self.Q, dtype=float, copy=False)
        c = np.array(self.c, dtype=float, copy=False)

        # Row sums of Q: sum_j Q_ij
        row_sums = Q.sum(axis=1)

        # Local fields h_i = -1/2 * (c_i + sum_j Q_ij)
        h = -0.5 * (c + row_sums)

        # Couplings J_ij = Q_ij / 2 for i != j, J_ii = 0
        J = np.zeros_like(Q, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                val = Q[i, j] / 2.0
                J[i, j] = val
                J[j, i] = val  # keep symmetric

        # Constant term: use the relation at z = (+1, ..., +1)
        # For z = +1, we have x = (1 - z)/2 = 0, so:
        #   E_Q(z_all_ones) = offset
        # and also:
        #
        #   offset = const + sum_i h_i + sum_{i<j} J_ij
        #
        # hence:
        #   const = offset - sum_i h_i - sum_{i<j} J_ij
        sum_J_upper = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sum_J_upper += J[i, j]

        const = float(self.offset - h.sum() - sum_J_upper)

        return h, J, const


    # ==========================
    # TRANSFORMATIONS / UTILITIES
    # ==========================

    def rescale(self, factor: float) -> None:
        """
        Rescale all energy coefficients by a constant factor.

            Q      <- factor * Q
            c      <- factor * c
            offset <- factor * offset

        Typical use cases:
        ------------------
        - adapt coefficients to hardware-specific ranges,
        - normalize energy scales across different instances,
        - perform sensitivity analyses.

        Notes:
        ------
        - This method mutates the current object in place.
        - If preservation of the original QUBO is important,
          use copy() before rescaling.
        """
        pass

    def prune_small_terms(self, threshold: float) -> None:
        """
        Set very small coefficients in Q and c to zero.

        Parameters
        ----------
        threshold:
            Positive value such that any |coef| < threshold can be
            considered numerical noise and set to zero.

        Intended effects:
        -----------------
        - reduce numerical noise from accumulated floating-point
          operations (constraints, rescaling, etc.),
        - potentially sparsify the model (if many tiny coefficients
          appear), which can help some backends.

        Notes:
        ------
        - The exact policy (e.g. symmetrize Q before/after pruning,
          handle diagonal vs. off-diagonal separately) can be defined
          later.
        - This method mutates the QUBO in place.
        """
        pass
