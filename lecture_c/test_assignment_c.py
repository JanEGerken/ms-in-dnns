import numpy as np

import numpy_nn


def test_np_linear():
    np_linear = numpy_nn.NPLinear(10, 20)
    np.random.seed(0xDEADBEEF)

    weight = np.random.randn(20, 10)
    bias = np.random.randn(20)
    x = np.random.randn(2, 10)
    np_linear.W = weight
    np_linear.b = bias

    with np.load("test_targets.npz") as targets:
        target_output = targets["output"]
        target_weight_grad = targets["weight_grad"]
        target_bias_grad = targets["bias_grad"]
        target_x_grad = targets["x_grad"]
        target_new_weight = targets["new_weight"]
        target_new_bias = targets["new_bias"]

    output = np_linear.forward(x)
    assert (np.max(np.abs(output - target_output))) < 1e-10

    x_grad = np_linear.backward(np.ones((2, 20)))
    assert (np.max(np.abs(target_x_grad - x_grad))) < 1e-10
    assert (np.max(np.abs(target_weight_grad - np_linear.W_grad))) < 1e-10
    assert (np.max(np.abs(target_bias_grad - np_linear.b_grad))) < 1e-10

    np_linear.gd_update(lr=0.1)
    new_weight = np_linear.W
    new_bias = np_linear.b

    assert (np.mean(np.abs(new_weight - target_new_weight))) < 1e-10
    assert (np.mean(np.abs(new_bias - target_new_bias))) < 1e-10


def test_np_mse_loss():
    np.random.seed(0xDEADBEEF)
    preds = np.random.randn(2, 5)
    targets = np.random.randn(2, 5)
    loss = numpy_nn.NPMSELoss()
    loss_value = loss.forward(preds, targets)

    target_loss_value = np.array(3.14486884)
    assert (np.max(np.abs(loss_value - target_loss_value))) < 1e-6

    input_grad = loss.backward()
    target_input_grad = np.array(
        [
            [0.56968961, -0.03161349, -0.50655807, 0.05449492, 0.30289607],
            [0.25702433, 0.08164547, -0.36852467, 0.36044526, 0.49257118],
        ]
    )
    assert (np.max(np.abs(input_grad - target_input_grad))) < 1e-6


def test_np_relu():
    np.random.seed(0xDEADBEEF)
    x = np.random.randn(2, 5)

    np_relu = numpy_nn.NPReLU()
    output = np_relu.forward(x)
    input_grad = np_relu.backward(np.ones_like(x))

    target_output = np.array(
        [
            [0.7026896, 0.0, 0.0, 0.57912336, 0.37605309],
            [1.58798934, 0.48476396, 0.0, 1.66647251, 1.4170007],
        ]
    )

    assert (np.max(np.abs(output - target_output))) < 1e-6

    target_input_grad = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0]])
    assert (np.max(np.abs(input_grad - target_input_grad))) < 1e-6


if __name__ == "__main__":
    test_np_linear()
    test_np_mse_loss()
    test_np_relu()
