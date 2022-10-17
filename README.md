# Automatic tests with Pytest
[![test-power-method](https://github.com/ttm02/ci-demo/actions/workflows/test-power-method.yml/badge.svg)](https://github.com/ttm02/ci-demo/actions/workflows/test-power-method.yml) ![coverage](./coverage.svg)
## What is this example about?

In this example we will execute some automatic tests for Python code implementing the [[power iteration method]](https://en.wikipedia.org/wiki/Power_iteration). This is an iterative algorithm for finding an approximation for the largest (in absolute value) eigenvalue $\lambda_\mathrm{max}$ of a diagonalisable matrix $A$. It also gives an approximate eigenvector $v$ corresponding to the solution of the eigenvalue equation $A v = \lambda_\mathrm{max} v$.

## Implementation and tests

### Power method implementation

The power iteration method is implemented in Python by makeing use of the [[Numpy]](https://github.com/numpy/numpy) library. The code can be found in `power_method.py`. For details of the implementation please refer to the source file.

### Tests

The tests are provided in the file `test_power_method.py`. The tests rely on test [[Pytest]](https://pytest.org/) test framework.

Two integration tests are implemented:

* Test the final result for the eigenvalue when using the eigenvalues to abort the iterations. The function 
  ```python
  def test_real_symmetric_iter_eigenvalues(matrix_file):
    # ...
  ```
  implements this test.

* Test the final result for the eigenvalue when using the eigenvectors to abort the iterations. The function
  ```python
  def test_real_symmetric_iter_eigenvector(matrix_file):
    # ...
  ```
  implements this test.

Both tests use real-valued symmetric matrices $A \in \mathbb{R}^{n \times n}$ with several different sizes. These matrices are contained in the directory `test_data`. 

## How to run the tests

Make sure to have Python version `>=3.9.x` installed on your system. If you do not have a corresponding Python version already installed you can install it with e.g. [[Miniconda]](https://docs.conda.io/en/latest/miniconda.html).

### Python environment

Use on the provided files `requirements.txt` or `environment.yml` to set up a Python [[virtual environment]](https://docs.python.org/3/tutorial/venv.html) or an [[Anaconda environment]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

#### Virtual environment

Open a terminal emulator and make sure Python is contained in the `PATH`:

```shell
$ which python3
/usr/bin/python3
```

Next, execute the following commands to establish the Python environment:

```shell
$ mkdir -vp pyenv           # Host venv in current workdir
$ python3 -m venv pyenv     # Create venv 
$ source pyenv/bin/activate # Acivate venv
$ pip install --upgrade pip && pip install -r requirements.txt
```

### Conda environment

Open a terminal emulator and make sure the `conda` utility is in the `PATH`. This should be the case if you have installed either [[Miniconda3]](https://docs.conda.io/en/latest/miniconda.html) or [[Anaconda3]](https://docs.conda.io/en/latest/miniconda.html).  Next, establish the Anaconda environment by running the following commands from the shell prompt:

```shell
$ conda env create --file environment.yml
```

This will create an environment named `AutomaticTesting` which can be activated like so:

```shell
$ source activate AutomaticTesting
```

### Running the tests

Pytest is needed to execute the tests. Hence make sure it has been properly installed:

```shell
$ python -c "import pytest; print(pytest.__version__)"
```

This should print the Pytest version number (7.1.2 in my case).

Running the test is as simple as

```shell
$ pytest test_power_method.py # Add -v for more verbose output.
```

If you just want to execute the tests for the eigenvalue criterion for aborting the iterations use:

```shell
$ pytest -k 'eigenvalues' test_power_method.py
```

The pattern `eigenvalues` will make Pytest select only the test function with a name matching this pattern. Conversely, if you want to limit yourself to the tests for the eigenvector criterion use the `eigenvectors` pattern:

```shell
$ pytest -k 'eigenvectors' test_power_method.py
```




