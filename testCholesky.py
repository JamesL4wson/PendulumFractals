import numpy as np

def cholesky(A):
    n = len(A)
    L = np.zeros_like(A, dtype=np.float64)
    for i in range(0, n):
        for j in range(0, i+1):
            sum_part = 0
            for k in range(0, j):
                sum_part += A[i, k] * A[j, k]

            if i == j:
                A[i, j] = np.sqrt(A[i, j] - sum_part)
            else:
                A[i, j] = (A[i, j] - sum_part) / A[j, j]
    return A

print(cholesky(np.array([[4, 2, 2],[2, 5, 1],[2, 1, 3]])))