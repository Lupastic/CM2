import numpy as np

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
], dtype=float)

b = np.array([18, 26, 34, 82], dtype=float)

def cramer(A, b):
    det_A = np.linalg.det(A)
    if det_A == 0:
        return "Can't solve"

    n = len(b)
    result = []
    for i in range(n):
        temp_A = A.copy()
        temp_A[:, i] = b
        det_temp_A = np.linalg.det(temp_A)
        result.append(float(det_temp_A / det_A))
    return result

def gauss(A, b):
    AB = np.hstack((A, b.reshape(-1, 1)))
    n = len(b)

    for i in range(n):
        max_row = np.argmax(abs(AB[i:, i])) + i
        AB[[i, max_row]] = AB[[max_row, i]]
        for j in range(i + 1, n):
            factor = AB[j, i] / AB[i, i]
            AB[j, i:] -= factor * AB[i, i:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (AB[i, -1] - np.dot(AB[i, i + 1:n], x[i + 1:])) / AB[i, i]
    return x

def is_dominant(A):
    for i in range(len(A)):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(len(A)) if i != j):
            return False
    return True

def jacobi(A, b, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new.copy()

    return x, max_iter

def gauss_seidel(A, b, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros(n)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1

    return x, max_iter

cramer_result = cramer(A, b)
gauss_result = gauss(A, b)

print("Cramers method:", cramer_result)
print("Gaussian method:", gauss_result)

if not is_dominant(A):
    print("matrix isnt diagonally dominant")
    jacobi_result, jacobi_iter = jacobi(A, b)
    gauss_seidel_result, gauss_seidel_iter = gauss_seidel(A, b)
    print(f"Jacobi method: {jacobi_result}, iterations: {jacobi_iter}")
    print(f"Gauss-Seidel method: {gauss_seidel_result}, iterations: {gauss_seidel_iter}")
