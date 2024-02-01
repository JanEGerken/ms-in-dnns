## Assignment A (15 Points)

### Task 1: Setting up the compute environment (5 Points)
Fork this repo into your own GitHub account, clone it to your local computer and install the virtual
environment as described in [README.md](README.md). All assignments should be done with this virtual
environment. If you think you need additional packages, ask for permission.

You can check if the packages can be imported into Python by running the script `test_assignment_a.py`.

On Canvas, submit
- A link to the forked repository
- The output of `pip3 freeze` (run inside the environment)
- The output of `git log`

### Task 2: Generating sequences and argparse (10 Points)
Write a python script `sequencer.py` which computes well-known mathematical sequences **iteratively, not recursively**, using Python lists. In terms of mathematics, you can use `math.sqrt` from the standard library, but do not import any other functions (e.g. `math.factorial` or anything from numpy etc). Implement a command-line interface using `argparse`. The `argparse` documentation is available [here](https://docs.python.org/3/library/argparse.html).

Your script should have the following interface: On the command line, it should accept a `--length` argument for the length of the computed sequence and a `--sequence` argument for the name of the sequence with possible values
- `fibonacci` for the [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_sequence)
- `prime` for the sequence of prime numbers
- `square` for the sequence of perfect squares
- `triangular` for the sequence of [triangular numbers](https://en.wikipedia.org/wiki/Triangular_number)
- `factorial` for the sequence of factorials

Your script should contain `main` function which takes as argument the parsed argparse namespace and returns the generated sequence as a list of integers. Your script should pass the test in `test_assignment_a.py`.
