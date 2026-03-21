"""Tests for analysis functions."""

import numpy as np
import pytest

from lwi_microbolometer_design.analysis import (
    spectral_angle_mapper,
    compute_distance_matrix,
    min_based_dissimilarity_score,
    group_based_dissimilarity_score,
)


class TestSpectralAngleMapper:
    """Test spectral angle mapper function."""

    def test_spectral_angle_mapper_identical_vectors(self):
        """Test that identical vectors have zero angle."""
        vector1 = np.array([1.0, 2.0, 3.0])
        vector2 = np.array([1.0, 2.0, 3.0])

        angle = spectral_angle_mapper(vector1, vector2)

        assert angle == 0.0

    def test_spectral_angle_mapper_orthogonal_vectors(self):
        """Test that orthogonal vectors have 90 degree angle."""
        vector1 = np.array([1.0, 0.0])
        vector2 = np.array([0.0, 1.0])

        angle = spectral_angle_mapper(vector1, vector2)

        assert abs(angle - 90.0) < 1e-10

    def test_spectral_angle_mapper_opposite_vectors(self):
        """Test that opposite vectors have 180 degree angle."""
        vector1 = np.array([1.0, 0.0])
        vector2 = np.array([-1.0, 0.0])

        angle = spectral_angle_mapper(vector1, vector2)

        assert abs(angle - 180.0) < 1e-10

    def test_spectral_angle_mapper_zero_vector_error(self):
        """Test that zero vectors raise an error."""
        vector1 = np.array([1.0, 2.0, 3.0])
        vector2 = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="zero magnitude"):
            spectral_angle_mapper(vector1, vector2)


class TestComputeDistanceMatrix:
    """Test distance matrix computation."""

    def test_compute_distance_matrix_basic(self):
        """Test basic distance matrix computation."""
        sensor_outputs = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ]
        )  # 2 basis functions, 3 substances

        distance_matrix = compute_distance_matrix(sensor_outputs, spectral_angle_mapper, axis=1)

        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (3, 3)  # 3x3 for 3 substances
        assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetric
        assert np.allclose(np.diag(distance_matrix), 0.0)  # Zero diagonal

    def test_compute_distance_matrix_single_substance(self):
        """Test distance matrix with single substance."""
        sensor_outputs = np.array(
            [
                [1.0],
                [2.0],
            ]
        )  # 2 basis functions, 1 substance

        distance_matrix = compute_distance_matrix(sensor_outputs, spectral_angle_mapper, axis=1)

        assert distance_matrix.shape == (1, 1)
        assert distance_matrix[0, 0] == 0.0


class TestScoringFunctions:
    """Test scoring functions."""

    def test_min_based_dissimilarity_score(self):
        """Test minimum-based dissimilarity score."""
        distance_matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )

        score = min_based_dissimilarity_score(distance_matrix=distance_matrix)

        assert score == 1.0  # Minimum off-diagonal value

    def test_group_based_dissimilarity_score(self):
        """Test group-based dissimilarity score."""
        distance_matrix = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 7.0, 8.0],
                [5.0, 7.0, 0.0, 2.0],
                [6.0, 8.0, 2.0, 0.0],
            ]
        )
        groups = [[0, 1], [2, 3]]  # Two groups

        score = group_based_dissimilarity_score(groups, distance_matrix=distance_matrix)

        # Should be mean of inter-group distances: (5+6+7+8)/4 = 6.5
        expected = (5.0 + 6.0 + 7.0 + 8.0) / 4.0
        assert abs(score - expected) < 1e-10

    def test_group_based_dissimilarity_score_insufficient_groups(self):
        """Test that insufficient groups raise an error."""
        distance_matrix = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        groups = [[0, 1]]  # Only one group

        with pytest.raises(ValueError, match="At least 2 groups"):
            group_based_dissimilarity_score(groups, distance_matrix=distance_matrix)
