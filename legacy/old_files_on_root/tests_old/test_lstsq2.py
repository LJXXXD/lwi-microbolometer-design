import numpy as np

# Step 1: Generate Synthetic Data
np.random.seed(0)  # For reproducibility

# Matrix A (10 vectors, each with 2 elements)
A = np.random.randint(1, 10, (4, 3))


# True weights for 5 samples (3 weights for each sample)
true_weights = np.array(
    [
        [0.7, 0.1, 0.2],
        [0.3, 0.2, 0.5],
        [0.6, 0.1, 0.3],
        [0.2, 0.4, 0.4],
        [0.3, 0.7, 0],
        [0.1, 0.5, 0.4],
        [0.9, 0, 0.1],
        [0.8, 0.1, 0.1],
        [0.4, 0.3, 0.3],
        [0.5, 0.3, 0.2],
    ]
)

# Generate noisy outputs for each sample
noise_level = 2
# noise = noise_level * np.random.normal(A.shape[0])
# print(noise)
# print('sum of noise:', sum(noise))
noisy_outputs = np.asarray(
    [A.dot(w) + noise_level * np.random.normal(A.shape[0]) for w in true_weights]
)
print("noisy_outputs", noisy_outputs.shape)


print("A:", A)
print("noisy_outputs:", noisy_outputs)

# sum_A = np.sum(A, axis=0)
# sum_w = np.sum(true_weights, axis=0)
# sum_o = [np.sum(o) for o in noisy_outputs]
sum_o = np.sum(noisy_outputs, axis=0)

# updated_A = sum_A - A
# updated_noisy_outputs = [sum_o - o for o in noisy_outputs]
updated_noisy_outputs = sum_o - noisy_outputs
print(updated_noisy_outputs)

# A = updated_A
noisy_outputs = updated_noisy_outputs


# print('A:', A)
# print('sum A:', sum_A)
print("sum_o:", sum_o)
# print('updated_A:', updated_A)
print("updated_noisy_outputs:", updated_noisy_outputs)


# Step 2: Solve for Weights and Get Residuals
estimated_weights = []
sum_squared_residuals = []
for y in noisy_outputs:
    w, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    estimated_weights.append(w)
    sum_squared_residuals.append(residuals if residuals.size > 0 else np.array([0]))

# Step 3: Analyze the Results
for i, (true_w, estimated_w, residual) in enumerate(
    zip(true_weights, estimated_weights, sum_squared_residuals)
):
    print(f"Sample {i + 1}:")
    print(f"True Weights: {true_w}")
    print(f"Estimated Weights: {estimated_w}")
    print(f"Sum of Squared Residuals: {residual}\n")

# Aggregate analysis of residuals
average_residuals = np.mean(sum_squared_residuals)
std_dev_residuals = np.std(sum_squared_residuals)

print("Average Sum of Squared Residuals Across All Samples:", average_residuals)
print("Standard Deviation of Sum of Squared Residuals Across All Samples:", std_dev_residuals)


estimated_weights = np.asarray(estimated_weights)
print(estimated_weights.shape)

size = len(true_weights)
np.eye(size)
coefficients = np.ones((size, size)) - np.eye(size)
print(coefficients)


w, r, _, _ = np.linalg.lstsq(coefficients, estimated_weights, rcond=None)
print(w)
