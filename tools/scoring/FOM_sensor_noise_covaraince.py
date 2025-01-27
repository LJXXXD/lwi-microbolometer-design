import numpy as np

def fom_sensor_noise_covariance(a_matrix):
    """
    Calculates the Figure of Merit (FOM) for sensor noise based on the covariance matrix 
    derived from the pseudo-inverse of the A matrix.

    Parameters:
    - a_matrix (2D array): The A matrix representing sensor outputs (M x N),
      where M is the number of basis functions, and N is the number of substances.

    Returns:
    - float: The Figure of Merit (FOM), calculated as the trace of the covariance matrix.
    """
    # Validate input type
    if not isinstance(a_matrix, np.ndarray):
        a_matrix = np.array(a_matrix)

    # Step 1: Compute the pseudo-inverse of the A matrix
    pseudo_inverse = np.linalg.pinv(a_matrix)

    # Step 2: Calculate the covariance matrix
    covariance_matrix = np.dot(pseudo_inverse, pseudo_inverse.T)

    # Step 3: Compute the Figure of Merit as the trace of the covariance matrix
    fom = np.trace(covariance_matrix)

    return fom