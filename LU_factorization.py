import numpy as np


def generate_a_b(n):
    return np.random.random((n, n)), np.random.random(n)


def LU_simple_factorization(a):
    n = len(a)
    U = a.copy()
    L = np.eye(n)
    for c in range(n):
        for r in range(c + 1, n):
            multiplier = U[r][c] / U[c][c]
            U[r] -= multiplier * U[c]
            L[r][c] = multiplier
    return L, U


def LU_Factorization_in_place(A):
    """
    Performs Factorization LU of A, overwriting A.

    Returns matrix LU, where lower triangle is L, diagonal nad upper triangle is U

    L also contains diagonal filled with ones, that needs to be remembered.

    :param A: np.array, dim=2
    :return: np.array, dim=2
    """
    n = len(A)
    for c in range(n):
        for r in range(c + 1, n):
            multiplier = A[r][c] / A[c][c]
            for k in range(c + 1, n):
                A[r][k] -= multiplier * A[c][k]
            A[r][c] = multiplier
    return A


def L_times_U_from_LU(LU):
    """
    O(n^2) memory 

    :param LU: 2d np.array, where lower triangle is L, diagonal and upper triangle is U
    :return: L @ U,          @ - matrix multiplication
    """
    n = len(LU)
    res = np.zeros((n, n))
    res[0] = LU[0]
    for L_row in range(1, n):
        for U_col in range(0, n):
            for k in range(min(L_row, U_col) + 1):
                if k == L_row:
                    res[L_row][U_col] += LU[k][U_col]
                else:
                    res[L_row][U_col] += LU[L_row][k] * LU[k][U_col]

    return res


def extract_LU(LU):
    """
    :param LU:  2d np.array, where lower triangle is L, diagonal and upper triangle is U
    :return: L, U
    """
    n = len(LU)
    U = np.triu(LU)
    L = np.tril(LU)
    for i in range(n):
        L[i][i] = 1
    return L, U


print(np.zeros((5,5)))
a = [[2,1,3], [4,-1,3], [-2, 5,5]]
a, _ = generate_a_b(6)
a = np.array(a, dtype=np.float64)

b = a.copy()
LU = LU_Factorization_in_place(a)
print(L_times_U_from_LU(LU))

L, U = extract_LU(LU)


