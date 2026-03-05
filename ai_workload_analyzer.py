import numpy as np
import time
import matplotlib.pyplot as plt

# Simulate naive AI workload
def naive_matrix_mult(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Simulate accelerated AI workload using NumPy
def accelerated_matrix_mult(A, B):
    return np.dot(A, B)

# Test different matrix sizes
matrix_sizes = [50, 100, 200, 300]
naive_times = []
accelerated_times = []

for n in matrix_sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Naive timing
    start = time.time()
    naive_matrix_mult(A, B)
    naive_times.append(time.time() - start)

    # Accelerated timing
    start = time.time()
    accelerated_matrix_mult(A, B)
    accelerated_times.append(time.time() - start)

# Plot results
plt.figure(figsize=(8,5))
plt.plot(matrix_sizes, naive_times, marker='o', label='Naive CPU')
plt.plot(matrix_sizes, accelerated_times, marker='o', label='Accelerated (NumPy)')
plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Execution Time (seconds)")
plt.title("AI/ML Workload Performance Analysis")
plt.legend()
plt.grid(True)
plt.show()