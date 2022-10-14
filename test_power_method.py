import numpy as np

import pytest
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy

import sys
import importlib
from pathlib import Path
from glob import glob

from power_method import power_method_iterations

# lower values for CI
MAX_ITERATIONS = 100
MAX_EXAMPLES = 10

RTOL = 1e-2

# one may wanmt to fix random seed for reproducible results
#np.random.seed(42)

# this helps for debugging

@pytest.mark.parametrize(
    'matrix_file',
    glob('./test_data/matrix_symmetric/mat*.txt')
)
def test_real_symmetric_iter_eigenvalues(matrix_file):
    A = np.loadtxt(matrix_file)

    eval_max_iter, _, iteration_count = power_method_iterations(
        A, maxit=MAX_ITERATIONS, criterion="eigenvalues")
    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")
    evals = np.linalg.eigvalsh(A)  # use different implementation to compute eigenvalues
    eval_max = evals[np.argmax(np.abs(evals))]
    # The choice of atol keeps the same ratio as the default values:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    assert np.allclose(eval_max, eval_max_iter, rtol=RTOL, atol=RTOL * 1e-3)  # do the comparision


@pytest.mark.parametrize(
    'matrix_file',
    glob('./test_data/matrix_symmetric/mat*.txt')
)
def test_real_symmetric_iter_eigenvectors(matrix_file):
    A = np.loadtxt(matrix_file)

    eval_max_iter, _, iteration_count = power_method_iterations(
        A, maxit=MAX_ITERATIONS, criterion="eigenvectors")
    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")
    evals = np.linalg.eigvalsh(A)
    eval_max = evals[np.argmax(np.abs(evals))]
    # The choice of atol keeps the same ratio as the default values:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    assert np.allclose(eval_max, eval_max_iter, rtol=RTOL, atol=RTOL * 1e-3)


@pytest.mark.parametrize(
    'matrix_file',
    glob('./test_data/matrix_symmetric/mat*.txt')
)
def test_multiplied_evec1(matrix_file):
    factor = 42
    A = np.loadtxt(matrix_file)

    eval_max_iter, e_vec, iteration_count = power_method_iterations(
        A, maxit=MAX_ITERATIONS, criterion="eigenvalues")

    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")

    e_vec_scaled = factor * e_vec
    Ax = np.dot(A, e_vec_scaled)

    ## Ax  should be lambda x, even if x is scaled by any factor
    assert np.allclose(np.dot(Ax, Ax) / np.dot(e_vec_scaled, e_vec_scaled),
                       eval_max_iter ** 2,
                       rtol=RTOL,
                       atol=RTOL * 1e-3)  # do the comparision


@settings(max_examples=MAX_EXAMPLES)
@given(
    # A
    hypothesis.extra.numpy.arrays(dtype=np.float64,
                                  shape=hypothesis.extra.numpy.array_shapes(min_dims=2, max_dims=2, min_side=2).filter(
                                      lambda x: x[0] == x[1]),
                                  elements=st.floats(allow_nan=False,
                                                     allow_infinity=False)),
    # factor
    st.floats(min_value=2, exclude_min=False, allow_nan=False, allow_infinity=False))
def test_multiplied_evec1_generated(A, factor):
    eval_max_iter, e_vec, iteration_count = power_method_iterations(
        A, maxit=MAX_ITERATIONS, criterion="eigenvalues")

    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")

    e_vec_scaled = factor * e_vec
    Ax = np.dot(A, e_vec_scaled)

    ## Ax  should be lambda x, even if x is scaled by any factor
    assert np.allclose(np.dot(Ax, Ax) / np.dot(e_vec_scaled, e_vec_scaled),
                       eval_max_iter ** 2,
                       rtol=RTOL,
                       atol=RTOL * 1e-3)  # do the comparision


@pytest.mark.parametrize(
    'matrix_file',
    glob('./test_data/matrix_symmetric/mat*.txt')
)
def test_multiplied_evec2(matrix_file):
    factor = 42
    A = np.loadtxt(matrix_file)

    eval_max_iter, e_vec, iteration_count = power_method_iterations(
        A, rtol=1e-12, maxit=MAX_ITERATIONS, criterion="eigenvalues")
    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")
    e_vec_scaled = factor * e_vec

    ## Ax  should be lambda x, even if x is scaled by any factor
    a = np.dot(A, e_vec_scaled)
    b = eval_max_iter * e_vec_scaled
    print(a, b)
    assert np.allclose(np.dot(A, e_vec_scaled), eval_max_iter * e_vec_scaled,
                       rtol=RTOL, atol=RTOL * 1e-3)


@settings(max_examples=MAX_EXAMPLES)
@given(
    # A
    hypothesis.extra.numpy.arrays(dtype=np.float64,
                                  shape=hypothesis.extra.numpy.array_shapes(min_dims=2, max_dims=2, min_side=2).filter(
                                      lambda x: x[0] == x[1]),
                                  elements=st.floats(min_value=1, exclude_min=True, allow_nan=False,
                                                     allow_infinity=False)),
    # factor
    st.floats(min_value=2, exclude_min=False, allow_nan=False, allow_infinity=False))
def test_multiplied_evec2_generated(A, factor):
    eval_max_iter, e_vec, iteration_count = power_method_iterations(
        A, rtol=1e-12, maxit=MAX_ITERATIONS, criterion="eigenvalues")

    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")

    e_vec_scaled = e_vec * factor

    ## Ax  should be lambda x, even if x is scaled by any factor
    a = np.dot(A, e_vec_scaled)
    b = eval_max_iter * e_vec_scaled
    print(a, b)
    assert np.allclose(np.dot(A, e_vec_scaled), eval_max_iter * e_vec_scaled,
                       rtol=RTOL, atol=RTOL * 1e-3)


@pytest.mark.parametrize(
    'matrix_file',
    glob('./test_data/matrix_symmetric/mat*.txt')
)
def test_inverse_evec(matrix_file):
    factor = 1
    A = np.loadtxt(matrix_file)

    eval_max_iter, e_vec, iteration_count = power_method_iterations(
        A, maxit=MAX_ITERATIONS, criterion="eigenvalues")
    if iteration_count == MAX_ITERATIONS:
        pytest.skip("Not performed sufficient iterations for satisfying result")

    try:
        A_Inv = np.linalg.inv(A)
        Ax = A_Inv.dot(e_vec)
        lambdaX = (1 / eval_max_iter) * e_vec
        # test if
        # A^-1 * x = 1/lambda * x
        assert np.allclose(np.dot(e_vec, e_vec),
                           np.dot(Ax, Ax) * eval_max_iter ** 2,
                           rtol=RTOL,
                           atol=RTOL * 1e-3)  # do the comparision

    except np.linalg.LinAlgError:
        pytest.skip("A has no inverse")


def test_criterion():
    A = np.asarray([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        eval_max_iter, _, _ = power_method_iterations(A, maxit=MAX_ITERATIONS, criterion="iterations")


def test_invalid():
    A = np.asarray([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        eval_max_iter, _, _ = power_method_iterations(A, maxit=MAX_ITERATIONS, criterion="eigenvalues")


def test_zero_relaxed():
    A = np.asarray([[0, 0], [0, 0]])

    eval_max_iter, _, _ = power_method_iterations(A, maxit=MAX_ITERATIONS, criterion="eigenvalues")

    assert np.allclose(0, eval_max_iter, rtol=RTOL, atol=RTOL * 1e-3)


def test_zero_strict():
    A = np.asarray([[0, 0], [0, 0]])

    eval_max_iter, _, _ = power_method_iterations(A, maxit=MAX_ITERATIONS, criterion="eigenvalues")

    assert eval_max_iter == 0 or eval_max_iter == 0 + 0j
