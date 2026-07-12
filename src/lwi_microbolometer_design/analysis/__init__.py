"""Analysis module for distance metrics, scoring functions, and robustness evaluation."""

# Dissimilarity scoring
from .dissimilarity_scoring import (
    group_based_dissimilarity_score,
    min_based_dissimilarity_score,
)

# Distance matrix computation
from .distance_matrix import compute_distance_matrix

# Elite clustering on optimal-pairing distances
from .elite_clustering import (
    family_labels_from_optimal_pairing_d,
    graph_component_labels_from_distance_matrix,
    graph_single_linkage_from_distance_matrix,
    optimal_pairing_distance_matrix,
)

# Distance metrics
from .distance_metrics import spectral_angle_mapper

# Optimal pairing distance
from .optimal_pairing_distance import calculate_optimal_pairing_distance

# Robustness evaluation
from .robustness import (
    ConditionLabel,
    RobustnessResult,
    align_nominal_fitness_to_training_scene,
    evaluate_archive_robustness,
    evaluate_elite_robustness,
    evaluate_solutions_robustness,
    find_nominal_scene_index,
    summarise_robustness,
)

# VAT / CLODD analysis
from .clodd import (
    CLODDResult,
    add_clodd_partition_lines,
    clodd_objective,
    clodd_partition,
    eedge,
    esquare,
    labels_from_cuts,
    map_labels_to_original_order,
    normalize_ordered_dissimilarity,
)
from .vat import ivat_transform, vat_reorder

__all__ = [
    # Dissimilarity scoring
    "group_based_dissimilarity_score",
    "min_based_dissimilarity_score",
    # Distance matrix computation
    "compute_distance_matrix",
    # Elite clustering (optimal-pairing D)
    "family_labels_from_optimal_pairing_d",
    "graph_component_labels_from_distance_matrix",
    "graph_single_linkage_from_distance_matrix",
    "optimal_pairing_distance_matrix",
    # Distance metrics
    "spectral_angle_mapper",
    # Optimal pairing distance
    "calculate_optimal_pairing_distance",
    # Robustness evaluation
    "ConditionLabel",
    "RobustnessResult",
    "align_nominal_fitness_to_training_scene",
    "evaluate_archive_robustness",
    "evaluate_elite_robustness",
    "evaluate_solutions_robustness",
    "find_nominal_scene_index",
    "summarise_robustness",
    # VAT / CLODD analysis
    "CLODDResult",
    "add_clodd_partition_lines",
    "clodd_objective",
    "clodd_partition",
    "eedge",
    "esquare",
    "ivat_transform",
    "labels_from_cuts",
    "map_labels_to_original_order",
    "normalize_ordered_dissimilarity",
    "vat_reorder",
]
