"""Implement the power method for finding the largest eigenvalue of a matrix."""

import numpy as np


def power_method_iterations(A, rtol=1e-5, maxit=10000, criterion="eigenvalues"):
    b_k = np.random.random(A.shape[0]) + 1j * np.random.random(A.shape[0])
    b_k_old = np.zeros_like(b_k)

    # some edge cases:
    if np.count_nonzero(A) == 0:
        # for 0 matrix: all nonzero vectors are valid, eigenvalue=0
        return 0, b_k, 0
    if len(A.shape) != 2:
        raise ValueError("Error, this function is only designed for 2D Arrays")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Error, not a Quadratic matrix, no eigenvectors defined")

    eval, eval_old = -10000, 10000
    numiter = 0

    for _ in range(maxit):
        eval_old = eval
        b_k_old[:] = b_k
        # update eigenvector
        b_k[:] = np.dot(A, b_k_old)
        # normalise since A is not necessarily unitary
        b_k_norm = np.linalg.norm(b_k)
        b_k /= b_k_norm
        # this line needs some changes (conj()) in case matrix/vectors are complex-valued
        eval = np.linalg.multi_dot([b_k.conj(), A, b_k]) / np.dot(b_k.conj(), b_k)
        numiter += 1
        # test if old and new vector are close
        if criterion == "eigenvectors":
            if np.allclose(b_k, b_k_old, rtol=rtol, atol=rtol * 1e-3):
                break
        elif criterion == "eigenvalues":
            if np.allclose(eval, eval_old, rtol=rtol, atol=rtol * 1e-3):
                break
        else:
            raise ValueError("ERROR - invalid value for 'criterion'.")

    print("Converged after {} iterations.".format(numiter))

    return eval, b_k, numiter
