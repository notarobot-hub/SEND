import numpy as np
from scipy.integrate import quad

# Power Method for approximating the largest singular value
def power_method(X, num_iterations=1500, tol=1e-6):
    b = np.random.rand(X.shape[1])
    b = b / np.linalg.norm(b)  # Normalize initial vector
    for _ in range(num_iterations):
        b_new = np.dot(X.T, np.dot(X, b))
        b_new_norm = np.linalg.norm(b_new)
        b_new /= b_new_norm
        if np.linalg.norm(b_new - b) < tol:
            break
        b = b_new
    sigma_max_approx = np.linalg.norm(np.dot(X, b_new))
    return sigma_max_approx

# Compute c_m coefficients
def compute_c_m(m, num_points=500):
    integrand = lambda x: (np.log(x) * np.cos(m * x)) / np.sqrt(1 - x**2)
    c_m, _ = quad(integrand, 0, 1, limit=num_points)
    return c_m * (1 / np.pi)

# Compute T_prime_m matrix
def compute_T_prime_m(X_normalized, limit, Z):
    K, Nz = Z.shape
    T = [[None for _ in range(Nz)] for _ in range(limit + 1)]
    for j in range(Nz):
        T[0][j] = Z[:, j]
        T[1][j] = X_normalized.T @ (X_normalized @ Z[:, j])
    for m in range(2, limit + 1):
        for j in range(Nz):
            T[m][j] = 2 * (X_normalized.T @ (X_normalized @ T[m-1][j])) - T[m-2][j]
    return T

# Compute d_m values
def compute_dm(T, Z, m):
    K, Nz = Z.shape
    dm = 0
    for j in range(Nz):
        dm += Z[:, j].T @ T[m][j]
    return dm / (K * Nz)

# Compute eigenscore based on Chebyshev approximation
def compute_eigenscore(X, limit):
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    X_normalized = (X - X_mean) / X_std
    sigma_max = power_method(X_normalized)
    X_normalized /= sigma_max

    c_m_values = np.array([compute_c_m(m) for m in range(1, limit + 1)])
    K = X.shape[1]
    Nz = 200
    Z = np.random.randn(K, Nz)
    T_prime_m_values = compute_T_prime_m(X_normalized, limit, Z)
    d_m_values = np.array([compute_dm(T_prime_m_values, Z, m) for m in range(1, limit + 1)])
    eigenscore = np.sum(d_m_values * c_m_values)
    return eigenscore

# Regular eigenscore method
def regular_eigenscore(X, alpha=1e-5):
    d, K = X.shape
    Jd = np.eye(d) - (1 / d) * np.ones((d, d))
    covariance_matrix = X.T @ Jd @ X
    regularized_cov_matrix = covariance_matrix + alpha * np.eye(K)
    eigenvalues, _ = np.linalg.eigh(regularized_cov_matrix)
    logdet = np.sum(np.log(eigenvalues))
    eigenscore = logdet / K
    return eigenscore

# Test correlation between the two methods
def test_correlation():
    X = np.random.randn(3048, 20
    )  # Generate random data
    limit = 100  # Set Chebyshev limit

    eigenscore1 = compute_eigenscore(X, limit)
    eigenscore2 = regular_eigenscore(X)

    print(f"Eigenscore (Chebyshev approximation): {eigenscore1}")
    print(f"Eigenscore (Regular method): {eigenscore2}")

    return eigenscore1, eigenscore2

# Normalize and compute correlation for multiple inputs
from scipy.stats import pearsonr

def compute_correlation(input_pairs):
    eigenscores1 = []
    eigenscores2 = []

    for eigenscore1, eigenscore2 in input_pairs:
        eigenscores1.append(eigenscore1)
        eigenscores2.append(eigenscore2)

    # Compute Pearson correlation for the collected lists
    correlation, _ = pearsonr(eigenscores1, eigenscores2)
    print(f"Correlation between the two methods: {correlation}")

    return correlation

corr_total = 0
for i in range(10):
    input_pairs = [
        test_correlation() for _ in range(10)
    ]
    corr = compute_correlation(input_pairs)
    del input_pairs
    corr_total += corr
    
print(f"Average correlation: {corr_total / 10}")


# Run the test

