"""CMA-ME emitters for quality-diversity optimisation.

Emitters wrap CMA-ES instances and adapt their search distributions based
on **archive improvement** rather than raw fitness, which is the core
mechanism that distinguishes CMA-ME from standard CMA-ES.

References
----------
Fontaine, M. C., et al. (2020). "Covariance Matrix Adaptation for the
Rapid Illumination of Behavior Space." *GECCO 2020*.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cma
import numpy as np

from .normalization import UnitCubeScaler

logger = logging.getLogger(__name__)


class EmitterBase(ABC):
    """Abstract base class for CMA-ME emitters.

    An emitter maintains an internal search distribution and generates
    batches of candidate solutions.  After evaluating candidates against
    the archive, the caller provides per-candidate **improvement** values
    so the emitter can update its distribution toward archive-improving
    regions of the search space.

    Subclasses must implement :meth:`ask`, :meth:`tell`, :meth:`restart`,
    and the :attr:`batch_size` / :attr:`converged` properties.
    """

    @abstractmethod
    def ask(self) -> list[np.ndarray]:
        """Generate a batch of candidate solutions.

        Returns
        -------
        list[np.ndarray]
            Candidate chromosomes sampled from the emitter's distribution.
        """

    @abstractmethod
    def tell(
        self,
        solutions: list[np.ndarray],
        improvements: list[float],
        fitnesses: list[float],
    ) -> None:
        """Update the search distribution using archive-improvement ranking.

        Parameters
        ----------
        solutions : list[np.ndarray]
            Candidates previously returned by :meth:`ask`.
        improvements : list[float]
            Per-candidate archive improvement (>0 means the candidate was
            inserted into the archive).
        fitnesses : list[float]
            Raw fitness values (used as a fallback ranking when no candidate
            improves the archive).
        """

    @abstractmethod
    def restart(self, x0: np.ndarray) -> None:
        """Restart the emitter centred on a new solution.

        Parameters
        ----------
        x0 : np.ndarray
            New mean for the search distribution (typically a random elite).
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of candidates generated per :meth:`ask` call."""

    @property
    @abstractmethod
    def converged(self) -> bool:
        """Whether the emitter should be restarted."""


class OptimizingEmitter(EmitterBase):
    r"""CMA-ES emitter that ranks candidates by archive improvement.

    Candidates are sampled from :math:`\mathcal{N}(\mathbf{m}, \sigma^2 C)`
    where the distribution parameters are adapted via CMA-ES.  Crucially,
    the ranking supplied to the CMA-ES update is based on each candidate's
    **contribution to the MAP-Elites archive** (discovering a new bin or
    improving an existing bin's fitness), *not* raw fitness alone.

    When no candidate in a batch improves the archive, raw fitness is used
    as a fallback to keep the distribution moving toward promising regions.

    Parameters
    ----------
    x0 : np.ndarray
        Initial distribution mean (typically an elite's chromosome).
    sigma0 : float
        Initial step-size in normalized unit-cube coordinates.
    bounds_low : list[float]
        Lower bound per gene.
    bounds_high : list[float]
        Upper bound per gene.
    batch_size : int | None
        CMA-ES population size.  ``None`` uses the library default
        :math:`4 + \lfloor 3 \ln n \rfloor`.
    restart_patience : int
        Restart after this many consecutive non-improving generations.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        bounds_low: list[float],
        bounds_high: list[float],
        batch_size: int | None = None,
        restart_patience: int = 30,
        seed: int | None = None,
    ) -> None:
        self._scaler = UnitCubeScaler.from_bounds(bounds_low, bounds_high)
        self._unit_lower = np.zeros(self._scaler.dimension, dtype=float).tolist()
        self._unit_upper = np.ones(self._scaler.dimension, dtype=float).tolist()
        self._batch_size_cfg = batch_size
        self._sigma0 = sigma0
        self._seed = seed
        self._restart_patience = restart_patience
        self._no_improvement_gens = 0
        self._total_restarts = 0
        self._last_ask_normalized: list[np.ndarray] | None = None
        self._create_cma(x0)

    def _create_cma(self, x0: np.ndarray) -> None:
        """Instantiate (or re-instantiate) the internal CMA-ES solver."""
        normalized_x0 = self._scaler.normalize(x0)
        opts: dict = {
            "bounds": [self._unit_lower, self._unit_upper],
            "verbose": -9,
            "tolfun": 1e-11,
            "tolx": 1e-11,
        }
        if self._batch_size_cfg is not None:
            opts["popsize"] = self._batch_size_cfg
        if self._seed is not None:
            opts["seed"] = self._seed
            self._seed += 1
        self._cma = cma.CMAEvolutionStrategy(normalized_x0.tolist(), self._sigma0, opts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of candidates generated per :meth:`ask` call."""
        return self._cma.popsize

    @property
    def total_restarts(self) -> int:
        """Cumulative restart count for this emitter."""
        return self._total_restarts

    def ask(self) -> list[np.ndarray]:
        """Sample candidates from the current CMA-ES distribution.

        Returns
        -------
        list[np.ndarray]
            ``batch_size`` candidate chromosomes.
        """
        self._last_ask_normalized = [np.asarray(x, dtype=float) for x in self._cma.ask()]
        return [self._scaler.denormalize(x) for x in self._last_ask_normalized]

    def tell(
        self,
        solutions: list[np.ndarray],
        improvements: list[float],
        fitnesses: list[float],
    ) -> None:
        """Update CMA-ES parameters using improvement-based ranking.

        Candidates that improved the archive are ranked above those that
        did not.  Among improvers, higher improvement earns a better rank.
        When **no** candidate improved the archive in this batch, raw
        fitness is used as a fallback so the distribution keeps adapting.

        Parameters
        ----------
        solutions : list[np.ndarray]
            Candidates previously returned by :meth:`ask`.
        improvements : list[float]
            Per-candidate archive improvement.
        fitnesses : list[float]
            Raw fitness values.
        """
        has_improvement = any(imp > 0 for imp in improvements)

        if has_improvement:
            self._no_improvement_gens = 0
            max_imp = max(improvements)
            objectives = [-imp if imp > 0 else (max_imp + 1.0) for imp in improvements]
        else:
            self._no_improvement_gens += 1
            objectives = [-f for f in fitnesses]

        if self._last_ask_normalized is not None and len(self._last_ask_normalized) == len(
            solutions
        ):
            normalized_solutions = [x.tolist() for x in self._last_ask_normalized]
        else:
            normalized_solutions = [self._scaler.normalize(s).tolist() for s in solutions]
        self._cma.tell(normalized_solutions, objectives)
        self._last_ask_normalized = None

    @property
    def converged(self) -> bool:
        """Whether the emitter's CMA-ES has converged or stalled."""
        if self._cma.stop():
            return True
        return self._no_improvement_gens >= self._restart_patience

    def restart(self, x0: np.ndarray) -> None:
        """Restart the emitter with a new mean.

        Parameters
        ----------
        x0 : np.ndarray
            New distribution mean (typically a random archive elite).
        """
        self._no_improvement_gens = 0
        self._total_restarts += 1
        self._last_ask_normalized = None
        logger.debug("Emitter restart #%d", self._total_restarts)
        self._create_cma(x0)
