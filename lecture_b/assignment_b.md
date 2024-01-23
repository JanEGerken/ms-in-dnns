# Assignment B (60 Points)

## 1. Launch a job on Google Cloud (10 Points)

Launch the `hello_world` program as a job on Google Cloud both as a script and as a package. You can find instructions in the README. Submit the terminal output of both launches in Canvas. If your GPU quota request has not been approved before the deadline, run the jobs on the CPU (i.e. set `N_GPUS` in the launch script to 0) and submit also a screenshot of the confirmation of your request.

## 2. Overfitting and regularization (50 Points)

In this exercise, you can see for yourself how a too flexible model can overfit on training data noise, reproducing some plots from Lecture 2. Your final script has to pass the tests in [test_assignment_b.py](test_assignment_b.py) to be considered for grading.

### 2.a Basic plotting in Matplotlib (5 Points)

Write a Python script `poly_fitting.py` which plots a sine between 0 and $2\pi$ and 15 training data points, reproducing the plots from Lecture 2. The noise is sampled from a Gaussian distribution with a standard deviation of 0.1 and the data points are uniformly distributed in $[0,2\pi]$. Also generate 10 test data points from the same distribution. You can find some basic examples of how to use Matplotlib [here](https://matplotlib.org/stable/plot_types/index.html).

### 2.b Polynomial regression (10 Points)

Add a function `fit_poly` to your script which performs polynomial regression **by implementing the closed-form solution for linear regression discussed in Lecture 2 in NumPy**. It should accept the following arguments
- A numpy array `x_train` of shape `(N,)` with the x-values of the training data
- A numpy array `y_train` of shape `(N,)` with the y-values of the training data
- An integer `k` for the order of the polynomial

where `N` is the number of training samples. This function should return a numpy array of shape `(1, k+1)` with the weights of the fitted polynomial. Using your function, fit a third order polynomial to your training data and plot it into the same plot as the sine and the training data.

Write a function `mse_poly` which returns the mean squared error of a fit polynomial on test data and has the arguments
- A numpy array `x` of shape `(N,)` with the x-values of the test data
- A numpy array `y` of shape `(N,)` with the y-values of the test data
- A numpy array `W` of shape `(1, k)` with the weights of the polynomial

Compute the MSE of the polynomial you have fitted and add it to the legend of the plot.

### 2.c Overfitting (5 Points)

Now, we want to study overfitting in this context. To this end, expand the domain of the data to $[0,4\pi]$ and generate 15 training- and 10 test data points in this interval.

1. Fit polynomials with $k=1,\dots,15$ and plot their MSEs against k with a logarithmic MSE-axis. Run your script a few times with different random seeds (if you have not set the seed explicitly, this means that you can just run your script several times) and see how the plot changes. Discuss what you observe.
2. Select a `k` which you think worked best across different seeds and plot the
   corresponding polynomial fit for one representative training data sample

### 2.d Ridge regression (15 Points)

The closed-form solution for ridge regression is given by
```math
W = Y^\top X \left(X^\top X + \lambda 1\right)^{-1}
```
in the notation of Lecture 2. Write a function `ridge_fit_poly` which performs ridge regression by implementing this closed-form solution. Your function should accept the following arguments
- A numpy array `x_train` of shape `(N,)` with the x-values of the training data
- A numpy array `y_train` of shape `(N,)` with the y-values of the training data
- An integer `k` for the order of the polynomial
- A float `lamb` for the regularization parameter
and return a numpy array of shape `(1, k+1)` with the weights of the fitted polynomial.

Now perform a two-dimensional hyperparameter optimization using grid search: For each combination of `k` in `list(range(1, 21))` and `lamb` in `10 ** np.linspace(-5, 0, 20)`, perform a ridge regression fit on the training data in $[0,4\pi]$ and compute the MSE on the test data. Then, plot the logarithms of the MSEs in a two-dimensional grid for the different values of `k` and `lamb` using `matplotlib.pyplot.imshow`.

Again, try a few different random seeds to get a feeling for the variance in your plot. Increase the amount of test data by a factor of 100 and see how the plot changes. Also increase the training data by a factor of 100 and see what happens. Discuss your results.

### 2.e Cross-validation (15 Points)

Write a function `perform_cv` to evaluate a combination of hyperparameters using cross-validation. Your function should accept these arguments:
- A numpy array `x` of shape `(N,)` with the x-values of the available data
- A numpy array `y` of shape `(N,)` with the y-values of the available data
- An integer `k` for the order of the polynomial
- A float `lamb` for the regularization parameter
- An integer `folds` which is a divisor of `N` for the number of folds to use

and return the CV estimate of the MSE.

Now choose a set of hyperparameters (i.e. order of the polynomial and regression parameter) which performs best on the test data. Generate a dataset of `120` samples and evaluate the hyperparameters using cross-validation with all fold numbers which are divisors of `120`. Repeat this 100 times with new datasets and compute the mean and standard deviation over draws for each fold number. Then plot the mean cross-validated MSE against the number of folds, together with a band of + and - its standard deviation. For the lower boundary, clip the values to zero since the MSE is positive. Observe how mean and standard deviation behave as you increase the number of folds and explain your observations.
