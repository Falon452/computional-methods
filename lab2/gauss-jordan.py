import numpy as np


def get_relative_error_from_result(result, actual, dtype=np.float64):
    return dtype(np.absolute(dtype(result)-actual)/actual)


def generate_a_b(n):
    return np.random.random((n, n)), np.random.random(n)


def gauss_jordan(a, b):
    def scale(a, b):
        new_a = []
        new_b = []

        for i, row in enumerate(a):
            max = np.max(row)
            new_a.append(row / max)
            new_b.append(b[i] / max)
        return new_a, new_b

    a, b = scale(a, b)  # not to overwrite given arguments
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    n = len(a)

    for c in range(n):
        pivot_index = np.argmax(a[c:, c]) + c
        a[[c, pivot_index]] = a[[pivot_index, c]]  # swap rows
        b[c], b[pivot_index] = b[pivot_index], b[c]

        for r in range(n):
            if r != c:
                multiplier = a[r][c] / a[c][c]
                a[r] -= multiplier * a[c]
                b[r] -= multiplier * b[c]

    return b / np.diagonal(a)


def get_relative_gauss_jordan_errors(n_of_errors=3, shapes=(5, 10, 15)):
    n = len(shapes)
    assert n_of_errors == n, "len(shape) must be equal to n_of_errors"
    errors = []

    for i in range(n):
        a, b = generate_a_b(shapes[i])
        x1 = gauss_jordan(a, b)
        x2 = np.linalg.solve(a, b)
        errors.append(np.average(get_relative_error_from_result(x1, x2)))

    return errors


if __name__ == '__main__':
    np.set_printoptions(15)

    a, b = generate_a_b(5)
    print(f"numpy result {np.linalg.solve(a, b)}")
    print(f"gauss result {gauss_jordan(a, b)}")


    print(get_relative_gauss_jordan_errors())