import numpy as np
import matplotlib.pyplot as plt

# Define the wavelength range in nanometers
wavelengths = np.linspace(400, 700, 100)  # Range from 400nm (blue) to 700nm (red)

# Cone response curves for the human eye (approximate)
# These values are normalized, and you can adjust them for your specific needs.
# These are just example values.
cone_response_R = np.exp(-(wavelengths - 560) ** 2 / (2 * 30 ** 2))  # Red-sensitive (L-cones)
cone_response_G = np.exp(-(wavelengths - 530) ** 2 / (2 * 30 ** 2))  # Green-sensitive (M-cones)
cone_response_B = np.exp(-(wavelengths - 460) ** 2 / (2 * 30 ** 2))  # Blue-sensitive (S-cones)

# Normalize the response curves
cone_response_R /= np.max(cone_response_R)
cone_response_G /= np.max(cone_response_G)
cone_response_B /= np.max(cone_response_B)

# Define the piecewise function
piecewise_curve = np.piecewise(wavelengths, [wavelengths < 450, (wavelengths >= 450) & (wavelengths <= 600), wavelengths > 600], 
                               [lambda x: ((x - 400) / 50) ** 2, 1, lambda x: ((x - 700) / 100) ** 2])

# Create a plot
plt.figure(figsize=(6, 4))
plt.plot(wavelengths, cone_response_R, 'r--', label='Red Cone Response')
plt.plot(wavelengths, cone_response_G, 'g--', label='Green Cone Response')
plt.plot(wavelengths, cone_response_B, 'b--', label='Blue Cone Response')

# Add the piecewise curve (yellow, thicker, labeled as 'Imaginary Response')
plt.plot(wavelengths, piecewise_curve, 'y', label='Imaginary Response', linewidth=2)

# Customize the plot
plt.title('Human Eye Cone Response Curves and Piecewise Curve')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Response')
plt.legend()
# plt.grid(True)

# Show the plot
plt.show()
