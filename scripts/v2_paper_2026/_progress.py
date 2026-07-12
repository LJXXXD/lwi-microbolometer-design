"""Real-time tqdm hooks for PyGAD / AdvancedGA runs in the presentation suite."""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm


class PresentationGAProgressBar:
    """Use as ``on_generation`` callback; displays ETA, gen/s, and best fitness postfix."""

    __slots__ = ("_pbar",)

    def __init__(self, total_generations: int, desc: str) -> None:
        self._pbar = tqdm(
            total=total_generations,
            desc=desc,
            unit="gen",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.05,
        )

    def on_generation(self, ga_instance: Any) -> None:
        if hasattr(ga_instance, "best_solutions_fitness") and ga_instance.best_solutions_fitness:
            best = float(ga_instance.best_solutions_fitness[-1])
        else:
            best = float(np.max(ga_instance.last_generation_fitness))
        self._pbar.update(1)
        self._pbar.set_postfix(best=f"{best:.2f}", refresh=True)

    def close(self) -> None:
        self._pbar.close()
