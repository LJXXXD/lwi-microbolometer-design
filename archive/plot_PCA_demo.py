import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create synthetic 2D data
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
X = np.random.multivariate_normal(mean, cov, 100)
print(X.shape)

# Initialize PCA with 2 components
pca_2d = PCA(n_components=2)
pca_1d = PCA(n_components=1)

# Fit and transform the data
X_pca_2d = pca_2d.fit_transform(X)
X_pca_1d = pca_1d.fit_transform(X)

# Percentage of variance explained by each component
explained_variance_ratio_2d = pca_2d.explained_variance_ratio_
explained_variance_ratio_1d = pca_1d.explained_variance_ratio_

# Get the eigenvectors and eigenvalues from the PCA model
eigen_vectors = pca_2d.components_
eigen_values = pca_2d.explained_variance_

print("Explained Variance Ratios (2D):", explained_variance_ratio_2d)
print("Total Explained Variance (2D):", np.sum(explained_variance_ratio_2d))
print("Explained Variance Ratios (1D):", explained_variance_ratio_1d)
print("Total Explained Variance (1D):", np.sum(explained_variance_ratio_1d))

# Plot original data, PCA 2D, and PCA 1D
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.quiver(0, 0, eigen_vectors[0, 0] * eigen_values[0], eigen_vectors[0, 1] * eigen_values[0], angles='xy', scale_units='xy', scale=1, color='red', label='Eigenvec. 1')
plt.quiver(0, 0, eigen_vectors[1, 0] * eigen_values[1], eigen_vectors[1, 1] * eigen_values[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Eigenvec. 2')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(2, 2, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1])
plt.title("PCA Transformed Data (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.subplot(2, 2, 3)
plt.scatter(X_pca_1d, np.zeros_like(X_pca_1d))
plt.title("PCA Transformed Data (1D)")
plt.xlabel("Principal Component 1")

plt.subplot(2, 2, 4)
plt.scatter(np.zeros_like(X_pca_2d[:, 1]), X_pca_2d[:, 1])
plt.title("Projected onto Principal Component 2")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()
