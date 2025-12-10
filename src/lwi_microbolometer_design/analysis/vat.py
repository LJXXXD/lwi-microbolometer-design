"""
VAT (Visual Assessment of Tendency) and iVAT (improved VAT) algorithms.

This module provides functions for reordering and transforming distance matrices
to reveal cluster structure visually. VAT reorders the matrix to group similar
items together, while iVAT further enhances the visualization by transforming
distance values to make cluster boundaries clearer.
"""

import numpy as np


def vat_reorder(distance_matrix: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Reorders a given distance matrix using the VAT method.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix.

    Returns
    -------
    Tuple[np.ndarray, list]
        A tuple containing:
        - vat_matrix: Reordered distance matrix for cluster visualization
        - reorder: The reordering of indices applied to the distance matrix
    """
    n = distance_matrix.shape[0]

    # Find the most dissimilar pair (farthest apart)
    # Flatten matrix and find max index, then convert back to 2D coordinates
    # (mypy has issues with np.argmax axis=None parameter)
    max_idx = int(np.argmax(distance_matrix.flatten()))
    i, _j = np.unravel_index(max_idx, distance_matrix.shape)

    # Start with one of the most distant points
    reorder = [i]
    remaining = set(range(n))  # Remaining points to process

    # Iteratively find the next closest point to any selected point
    while remaining:
        next_index = min(remaining, key=lambda x: min(distance_matrix[x, p] for p in reorder))
        reorder.append(next_index)
        remaining.remove(next_index)

    # Apply the reordering to the distance matrix
    # Convert reorder list to array for np.ix_ (mypy requires arrays, not lists)
    reorder_array = np.array(reorder, dtype=np.int64)
    vat_matrix = distance_matrix[np.ix_(reorder_array, reorder_array)]

    return vat_matrix, reorder


def ivat_transform(vat_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the iVAT distance transform (path-based distances) from a VAT-reordered matrix.

    The input vat_matrix should already be reordered via VAT, meaning its rows and columns
    follow a minimum spanning tree-like sequence. This function replaces each direct distance
    with the "bottleneck" distance along the MST path—i.e., the maximum edge on that path.

    Algorithm steps:
    For each new row r (1..N-1):
        - Find j < r that yields the minimum distance in vat_matrix[r, :r].
          (This j is the MST edge that would connect r to the existing tree.)
        - Set iVat[r, j] to that distance (symmetric as well).
        - For each c < r, c != j:
          iVat[r, c] = max( vat_matrix[r, j], iVat[j, c] )
          (The path from r to c goes through j, so the cost is the maximum of
          the r->j edge and the already computed j->c path.)

    Parameters
    ----------
    vat_matrix : np.ndarray
        The VAT-reordered distance matrix, typically symmetrical and
        with zeros on the diagonal.

    Returns
    -------
    np.ndarray
        The iVAT-transformed distance matrix in the same VAT ordering.
    """
    n = vat_matrix.shape[0]
    ivat_matrix = np.zeros_like(vat_matrix)

    for r in range(1, n):
        # Find the closest connection j for row r among previously processed rows
        # np.argmin returns int64 scalar, convert to Python int for indexing
        j = int(np.argmin(vat_matrix[r, :r]))

        # Direct MST edge between r and j
        ivat_matrix[r, j] = vat_matrix[r, j]
        ivat_matrix[j, r] = ivat_matrix[r, j]

        # Update path-based distances to all other previous vertices c
        for c in range(r):
            if c != j:
                # Bottleneck distance: the worst edge on the path r->j->c
                ivat_matrix[r, c] = max(vat_matrix[r, j], ivat_matrix[j, c])
                ivat_matrix[c, r] = ivat_matrix[r, c]

    return ivat_matrix
