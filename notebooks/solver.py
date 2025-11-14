
# ---------------------------------------------------------------------------
# Solver interface and QAOA-based implementation
# ---------------------------------------------------------------------------


@dataclass
class SolverResult:
    """
    Container for the result returned by a Solver.

    Attributes
    ----------
    bitstring : np.ndarray
        Binary vector representing the chosen solution.
    energy : float
        Objective value (Hamiltonian expectation or best sample energy).
    extra : Dict[str, Any]
        Optional extra information (parameters, history, raw samples, etc.).
    """

    bitstring: np.ndarray
    energy: float
    extra: Dict[str, Any]


class BaseSolver:
    """
    Abstract base class for QUBO/Ising solvers.

    Concrete solvers (e.g., QAOASolver) should implement the `solve` method.
    """

    def solve(self, qubo: QUBO) -> SolverResult:
        """
        Solve the given QUBO instance and return the best solution found.

        Parameters
        ----------
        qubo : QUBO
            The QUBO model to be solved.

        Returns
        -------
        SolverResult
            The solution bitstring and associated energy.
        """
        raise NotImplementedError


class QAOASolver(BaseSolver):
    """
    QAOA-based solver using PennyLane.

    This class is responsible for:
        - building the QAOA circuit from the given Hamiltonian,
        - optimizing the variational parameters,
        - extracting the best bitstring from samples or expectation.
    """

    def __init__(
        self,
        p: int = 1,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        optimizer: Optional[Any] = None,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.p = p
        self.device_name = device
        self.shots = shots
        self.optimizer = optimizer  # e.g., qml.GradientDescentOptimizer()
        self.max_steps = max_steps
        self.seed = seed

    def solve(self, qubo: QUBO) -> SolverResult:
        """
        Solve the given QUBO using QAOA.

        Parameters
        ----------
        qubo : QUBO
            The QUBO model to be solved.

        Returns
        -------
        SolverResult
            Best bitstring and corresponding energy.
        """
        # TODO:
        #   - build Hamiltonian via qubo.to_pennylane_hamiltonian()
        #   - define QAOA circuit ansatz
        #   - define cost function (expectation value of H)
        #   - run classical optimizer for max_steps
        #   - extract best bitstring from the final state or from samples
        raise NotImplementedError("QAOA solver not implemented yet.")
