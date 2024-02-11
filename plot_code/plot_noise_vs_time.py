
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



noise_perc_list = [i * 0.05 for i in range(21)]
noise_perc_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 1]

test_aggregation = range(1, 1000)
# test_aggregation = np.asarray([1, 5, 25, 125, 625, 3125])
# test_aggregation = np.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

plt.figure()

np.random.seed(28)

for max_noise_perc in noise_perc_list:

    noise_over_aggregation = []
    for agg in test_aggregation:
        noise_perc = np.random.uniform(-max_noise_perc, max_noise_perc, size=agg)
        noise_perc = np.abs(np.mean(noise_perc, axis=0))
        noise_over_aggregation.append(noise_perc)

    # Transform the features to include polynomial terms up to degree 2
    poly_features = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly_features.fit_transform(np.asarray(test_aggregation).reshape(-1, 1))

    print(X_poly)

    # Create a linear regression model
    model = LinearRegression()
    # print(noise_over_aggregation)
    # Fit the model to the polynomial features
    model.fit(X_poly, noise_over_aggregation)

    plt.plot(test_aggregation, model.predict(X_poly))


    # plt.plot(test_aggregation, noise_over_aggregation)

plt.xlabel('Test Aggregation')
plt.ylabel(f'Noise in data (Volt)')
plt.title(f'Noise vs Test Aggregation')
# Set logarithmic scale on x-axis with base 5
plt.xscale('log', base=5)
plt.legend(noise_perc_list, title = "Sensor Noise (Volt)") 

# Display a legend
# plt.legend(noise_perc_list)

# Show the plot
plt.show()