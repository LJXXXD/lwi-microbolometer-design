"""Analysis module for distance metrics, scoring functions, and robustness evaluation."""

# Dissimilarity scoring
from .dissimilarity_scoring import (
    group_based_dissimilarity_score,
    min_based_dissimilarity_score,
)

# Distance matrix computation
from .distance_matrix import compute_distance_matrix

# Distance metrics
from .distance_metrics import spectral_angle_mapper

# Optimal pairing distance
from .optimal_pairing_distance import calculate_optimal_pairing_distance

# Robustness evaluation
from .robustness import (
    ConditionLabel,
    RobustnessResult,
    evaluate_archive_robustness,
    evaluate_elite_robustness,
    evaluate_solutions_robustness,
    summarise_robustness,
)

# VAT analysis
from .vat import ivat_transform, vat_reorder

__all__ = [
    # Dissimilarity scoring
    "group_based_dissimilarity_score",
    "min_based_dissimilarity_score",
    # Distance matrix computation
    "compute_distance_matrix",
    # Distance metrics
    "spectral_angle_mapper",
    # Optimal pairing distance
    "calculate_optimal_pairing_distance",
    # Robustness evaluation
    "ConditionLabel",
    "RobustnessResult",
    "evaluate_archive_robustness",
    "evaluate_elite_robustness",
    "evaluate_solutions_robustness",
    "summarise_robustness",
    # VAT analysis
    "ivat_transform",
    "vat_reorder",
]
