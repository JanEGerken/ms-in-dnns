# Assignment C (60 Points)

## 1. Polynomial regression in PyTorch (25 Points)

In this execrices, you will study the polynomial regression problem on the sine we considered in Assignment B again. However, instead of using the closed-form solution for linear regression, we want to experiment with optimizing the MSE loss using gradient descent in PyTorch.

### 1.a Gradient descend and initialization (10 Points)
Implement the polynomial regression problem from Assignment B in PyTorch. For this entire exercise, we will fix the order of the polyonmial to be 3 (i.e. four weights need to be determined). As training data use
```python
N_TRAIN = 15
SIGMA_NOISE = 0.1

torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE
```
Reimplement the closed-form solution for the polynomial regression in PyTorch. Then, find the best weights by minimizing the MSE loss using `torch.optim.SGD`. Starting from all weights set to one, find the learning rate which minimizes the final loss after 100 steps. Plot the loss against the number of training steps. Create a second plot with the ground truth sine, the training data, the exact solution and the optimized polynomial. Discuss what you find when changing the learning rate.

### 1.b Different initialization and optimization algorithms (15 Points)

You will see that it is not easy to solve this optimization problem using gradient descent. Explain why (hint: compute the Hessian of the loss). One way to make the problem easier is to initialize the weights as $0.1^k$ for the $x^k$ term. Another strategy to obtain a better solution is to use gradient descent with momentum. Find a good combination of learning rate and momentum which works well for this problem and report the final loss after 100 optimization steps. Plot the loss against steps and the final optimized polynomial.

Next, try using the Adam optimizer to solve this problem. Find a good learning rate, plot the loss against steps and the final fit. Report the final loss.

PyTorch includes an implementation of the second-order [Limited-Memory BFGS algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) under `torch.optim.LBFGS` which computes an approximation to the inverse of the Hessian. Use this optimizer to find the best-fit polynomial, report the final loss and plot the loss against steps as well as the final fit. Discuss your results.

## 2. Neural network training in numpy (35 Points)

In this exercise, you will write a Python script `numpy_nn.py` which implements and trains a simple neural network in NumPy, illustrating what PyTorch does with its autograd system.

### 2.a Backpropagation in NumPy (20 Points)
Implement a linear layer similar to `torch.nn.Linear` in numpy. The linear layer should be a class `NPLinear` with attributes
- `W`, a 2d numpy array for the weight
- `b`, a 1d numpy array for the bias
- `W_grad`, a 2d numpy array for the gradient of `W`
- `b_grad`, a 1d numpy array for the gradient of `b` 

(plus any more which you might need), as well as methods
- `forward`, which accepts a batch of inputs of shape `(batch, in_channels)` and returns the output $Wx+b$ of the layer,
- `backward`, which accepts the gradient of the loss with respect to the output on a batch of inputs of shape `(batch, out_channels)` and sets `W_grad` and `b_grad` to the gradients of the loss w.r.t. `W` and `b`, summed over samples in the batch
- `gd_update` which accepts a learning rate (float) and performs a gradient descent step with that learning rate on the weights and biases

plus the constructor which should accept the number of input- and output channels. For the initalization of the parameters, use the same procedure that PyTorch uses (hint: check the implementation [here](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)).

Next, write a class `NPMSELoss` for the MSE loss in a similar fashion. It should have two methods
- `forward` which accepts prediction- and target arrays of shape `(batch, channels)` and returns the MSE loss averaged over samples
- `backward` which accepts no arguments and returns the gradient of the loss with respect to the perdictions.

Your gradients should be computed in such a way that `NPLinear.backward` on the gradient computed by the loss sets the gradients of the weights and biases to the correct values.

Finally, we need a nonlinearty. Implement ReLU in numpy as a class NPReLU which, as the loss, has two methods
- `forward` which accepts an array of inputs of shape `(batch, channels)` and returns the ReLU of that input
- `backward` which accepts an array of loss-gradients of shape `(batch, channels)` and returns the loss-gradient w.r.t. the input of the ReLU layer.

Use the averaging procedure which `torch.nn.MSELoss` uses when called with default arguments.

### 2.b Neural network training in NumPy (15 Points)
Now use the layers you have created to build a neural network in NumPy. We will train the neural network to perform regression on the (noisy) sine-data we have used in the previous assignment. Create a class `NPModel` which implements a fully-connected neural network with ReLU nonlinearities. It should have three methods
- `forward` which accepts a batch of inputs of shape `(batch, 1)` (i.e. one input channel) and returns the prediction
- `backward` which accepts the gradient of the loss of shape `(batch, 1)` (i.e. one output channel) and sets the gradients of the weights and biases
- `gd_update` which accepts the learning rate and performs one step of gradients descent

Now train your neural network with the MSE-loss on data generatey by
```python
N_TRAIN = 100
N_TEST = 1000
SIGMA_NOISE = 0.1

np.random.seed(0xDEADBEEF)
x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE
```
For this problem, you can use full-batch training. Plot training- and validation loss logarithmically against epochs and plot the training data, ground truth function (sine) and your neural network prediction for a few select epochs during training to see how your neural network learns the data.

Optimize the depth and width of your network as well as the learning rate and number of training epochs to reach a validation loss of 0.035. Hint: You can use an exponentially decaying learning rate schedule by multiplying your learning rate after each epoch with a factor in $(0,1)$.
