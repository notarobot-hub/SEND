import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import svd
import time
import csv
from sensitivity.run_with_temperature import *
from scipy.stats import pearsonr


def power_method(X, num_iterations=500, tol=1):
    b = np.random.rand(X.shape[1])
    for _ in range(num_iterations):
        b_new = np.dot(X.T, np.dot(X, b))
        b_new_norm = np.linalg.norm(b_new)
        b_new /= b_new_norm
        if np.linalg.norm(b_new - b) < tol:
            break
        b = b_new
    sigma_max_approx = np.linalg.norm(np.dot(X, b_new))
    return sigma_max_approx

def compute_c_m(m, num_points=500):
    integrand = lambda x: (np.log(x) * np.cos(m * x)) / np.sqrt(1 - x**2)
    c_m, _ = quad(integrand, 0, 1, limit=num_points)
    return c_m * (2 / np.pi)

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

def compute_dm(T, Z, m):
    K, Nz = Z.shape
    dm = 0
    for j in range(Nz):
        dm += Z[:, j].T @ T[m][j]
    return dm / (K * Nz)

# def compute_eigenscore(X, limit):
#     X_mean = np.mean(X, axis=0, keepdims=True)
#     X_std = np.std(X, axis=0, keepdims=True)
#     X_normalized = (X - X_mean) / X_std
#     sigma_max = power_method(X_normalized)
#     X_normalized /= sigma_max

#     c_m_values = np.array([compute_c_m(m) for m in range(1, limit + 1)])
#     K = X.shape[1]
#     Nz = 20
#     Z = np.random.randn(K, Nz)
#     T_prime_m_values = compute_T_prime_m(X_normalized, limit, Z)
#     d_m_values = np.array([compute_dm(T_prime_m_values, Z, m) for m in range(1, limit + 1)])
#     eigenscore = np.sum(d_m_values * c_m_values)
#     return eigenscore

def regular_eigenscore(X, alpha=1e-5):
    d, K = X.shape
    Jd = np.eye(d) - (1 / d) * np.ones((d, d))
    covariance_matrix = X.T @ Jd @ X
    regularized_cov_matrix = covariance_matrix + alpha * np.eye(K)
    eigenvalues, _ = np.linalg.eigh(regularized_cov_matrix)
    logdet = np.sum(np.log(eigenvalues))
    eigenscore = logdet / K
    return eigenscore

def manual():
    # create a matrix with eigenvalues only between 0 and 1
    X = np.random.randn(10, 10) * 0.5
    # print the eigenvalues spectrum 
    print(np.linalg.eigvals(X))

def ablation_study():
    row_values = [15000]
    col_values = [5000]
    limits = [10]

    results = []

    for rows in row_values:
        for cols in col_values:
            # using scikit create a positive defininte matrix
            X = np.random.randn(rows, cols)
            X = X @ X.T
            
            for limit in limits:
                start_time = time.time()
                eigenscore = compute_eigenscore(X, limit)
                print("the eigenscore is ", eigenscore)
                time_taken = time.time() - start_time

                start_time2 = time.time()
                eigenscore2, time_taken2 = regular_eigenscore(X)
                print("the regular eigenscore is ", eigenscore2)
                time_taken2 = time.time() - start_time2

                results.append((rows, cols, limit, time_taken, time_taken2, eigenscore, eigenscore2))
                print(f"Rows: {rows}, Cols: {cols}, Limit: {limit}, EES Time: {time_taken:.4f}, Regular Time: {time_taken2:.4f}")

    return results

def plot_comparison_of_times(results, filename='comparison_of_times.png'):
    row_values = sorted(set(r[0] for r in results))
    col_values = sorted(set(r[1] for r in results))
    limit = min(set(r[2] for r in results))  # Use the lowest moment number

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ees_times = np.array([r[3] for r in results if r[2] == limit]).reshape(len(row_values), len(col_values))
    regular_times = np.array([r[4] for r in results if r[2] == limit]).reshape(len(row_values), len(col_values))

    X, Y = np.meshgrid(col_values, row_values)

    ax1.plot_surface(X, Y, ees_times, cmap='viridis')
    ax1.set_title('EES Time (m = {})'.format(limit))
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')
    ax1.set_zlabel('Time (s)')

    ax2.plot_surface(X, Y, regular_times, cmap='plasma')
    ax2.set_title('Regular Time')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Rows')
    ax2.set_zlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_comparison_of_eigenscores(results, filename='comparison_of_eigenscores.png'):
    row_values = sorted(set(r[0] for r in results))
    col_values = sorted(set(r[1] for r in results))
    limits = sorted(set(r[2] for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for rows in row_values:
        for cols in col_values:
            eigenscores = [r[5] for r in results if r[0] == rows and r[1] == cols]
            ax.scatter([rows] * len(limits), eigenscores, label=f'Cols = {cols}', marker='o')

            for i in range(len(limits) - 1):
                ax.annotate('', xy=(rows, eigenscores[i+1]), xytext=(rows, eigenscores[i]),
                            arrowprops=dict(arrowstyle='<->', color='red'))

    ax.set_title('Comparison of Eigenscores for Different m Values')
    ax.set_xlabel('Rows')
    ax.set_ylabel('Eigenscore')
    ax.legend(title='Columns')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def print_table(results):
    print(f"{'Rows':<10}{'Cols':<10}{'Limit':<10}{'EES Time':<15}{'Regular Time':<15}{'EES Score':<15}{'Regular Score':<15}")
    for row in results:
        print(f"{row[0]:<10}{row[1]:<10}{row[2]:<10}{row[3]:<15.4f}{row[4]:<15.4f}{row[5]:<15.4f}{row[6]:<15.4f}")

def save_results_to_csv(results, filename='ablation_study_results.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rows', 'Cols', 'Limit', 'EES Time', 'Regular Time', 'EES Score', 'Regular Score'])
        for row in results:
            writer.writerow(row)

def llm_experiment(prompt, model_type):
    accelerator = Accelerator()
    model_name = model_type
    GUIDANCE = "You are my completion assistant. Continue the following prompt and complete it based on facts:  "
    input = GUIDANCE + prompt
    model_name = f"EleutherAI/pythia-{model_name}"
    embeddings = run_with_temperature(accelerator, model_name, input)
    # make embeddings symmetric by computing covariance matrix
    es = compute_eigenscore(embeddings, 100)
    es2 = regular_eigenscore(embeddings)
    return es, es2

def compute_correlation(prompts, model_type):
    eigenscores = []
    regular_eigenscores = []

    for prompt in prompts:
        es, es2 = llm_experiment(prompt, model_type)
        print(es, es2)
        eigenscores.append(es)
        regular_eigenscores.append(es2)

    correlation, _ = pearsonr(eigenscores, regular_eigenscores)
    return correlation