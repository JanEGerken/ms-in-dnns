# Assignment E: PyTorch Lightning and Equivariant Networks (60 Points)

## Task 1: CIFAR10 in Lightning (15 Points)

This exercise builds on the `SimpleCIFARNet` model provided in Assignment D. You will port this
model to a proper PyTorch Lightning package structure.

This task can be done either locally or on Google Cloud.

### 1.a Port to Lightning (10 Points)

Make the `SimpleCIFARNet` model into a `cifar10_net` package using Lightning, as discussed in
the lecture. You can start from the `income_net` package developed in the lecture which is available
in this directory. This means you should have the following directory structure:

```
cifar10_net
├── cifar10_net
│   ├── data.py
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── setup.py
```

In order to use your package while still developing it, you have to install it in your *venv* in
*editable* mode. For this, execute the following in the root of your package (i.e. in the folder
where the `setup.py` file is located):

```bash
pip install -e .
```

For more information, see the [Python Packaging User
Guide](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode).

Add the functionality to log the training- and validation accuracy as well as -loss during training
and separately for the best epoch (according to validation accuracy). Then train the model from
scratch for 60 epochs. Add the loss and validation plots in a WandB report.

### 1.b Confusion Matrix (5 Points)

Implement logging of the confusion matrix on the best epoch, similar to what you did using pure
PyTorch in the previous assignment. In particular:

- Use torchmetrics to compute the confusion matrix during the test step.
- Using a hook at the end of the test epoch, log the confusion matrix using a direct call to `wandb`
  (i.e. not `self.log`).
- One can log the computed confusion matrix without having wandb computing it from predictions and
  targets, which is apparent from the last lines of the implementation of
  `wandb.plot.confusion_matrix`
  [here](https://github.com/wandb/wandb/blob/6a211b19f02ee7c6b87b82eafd5789c4ba3739ec/wandb/plot/confusion_matrix.py#L82).
  Inspired by those lines, a working example can look like this:

  ```python
  class_names = self.trainer.datamodule.CLASS_NAMES
  data = []
  for i in range(10):
      for j in range(10):
          data.append([class_names[i], class_names[j], counts[i, j]])
  fields = {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"}
  conf_mat = wandb.plot_table(
      "wandb/confusion_matrix/v1",
      wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
      fields,
      {"title": "confusion matrix on best epoch"},
      split_table=True,
  )
  wandb.log({"best/conf_mat": conf_mat})
  ```

  Here, `counts` is a 2D array representing the entries of the non-normalized confusion matrix
  that is available from torchmetrics.

## Task 2: Equivariant Neural Networks (45 Points)

In this exercise, you will explore rotation equivariance using the
[escnn](https://github.com/QUVA-Lab/escnn) library and the
[MedMNIST](https://medmnist.com/) dataset. Medical images often have no canonical orientation,
making equivariant networks particularly valuable in this domain.

For Windows, we recommend WSL for this task. See the light tutorial in `README.md`.

This task can be done either locally or on Google Cloud.

### Setup

Install the required packages:

```bash
pip install escnn medmnist
```

Create a new package `medmnist_equivariant` with the same structure as `cifar10_net`. A
template for the package is provided in this directory. As always, install the package and its
dependencies in editable mode:

```bash
cd medmnist_equivariant
pip install -e .
```

If the installation fails, try:

```bash
sudo apt-get install -y gfortran python3.10-dev
```

We recommend using the **PathMNIST** dataset (colon pathology images, 9 classes, RGB 28x28) for this
exercise, but you may also experiment with other MedMNIST datasets like DermaMNIST.

### 2.a Baseline Model (10 Points)

Train a standard (non-equivariant) CNN baseline on PathMNIST:

- Implement a simple CNN with 3 convolutional blocks (conv -> ReLU -> pool), similar in spirit to
  SimpleCIFARNet but adapted for 28x28 images
- Wrap in a LightningModule with proper training/validation/test steps
- Train for 50 epochs and log metrics to WandB
- Report final test accuracy

### 2.b C4-Equivariant Model (20 Points)

Implement a C4-equivariant CNN using the escnn library. The C4 group consists of rotations by
multiples of 90 degrees.

Key escnn concepts:
- `gspaces.rot2dOnR2(N=4)` defines the C4 symmetry group
- `enn.FieldType` specifies how features transform under the group
- `enn.R2Conv` performs equivariant convolution
- `enn.GroupPooling` produces invariant features from equivariant ones

Your model should:
- Use `gspaces.rot2dOnR2(N=4)` for the symmetry group
- Build equivariant conv layers using `enn.R2Conv`
- Use `enn.GroupPooling` before the classifier to get rotation-invariant features
- Have a similar number of parameters to your baseline model

Train for 50 epochs with the same hyperparameters as the baseline and report test accuracy.

Useful resources:
- [escnn documentation](https://quva-lab.github.io/escnn/)
- [UvA Deep Learning Course - Geometric Deep Learning](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html)

### 2.c Data Augmentation Comparison (10 Points)

Compare four configurations to understand the relationship between data augmentation and
equivariance:

1. **Baseline CNN** without rotation augmentation
2. **Baseline CNN** with random rotation augmentation (0-360 degrees)
3. **Equivariant CNN** without rotation augmentation
4. **Equivariant CNN** with rotation augmentation

For each configuration, train for 50 epochs and report:
- Final test accuracy
- Training curves (loss and accuracy)

Create a comparison table in your WandB report.

### 2.d Analysis (5 Points)

Discuss in your WandB report:
- How does the equivariant model compare to the baseline in terms of accuracy?
- Does data augmentation help the equivariant model? Why or why not?
- What are the computational trade-offs (training time, memory usage)?
- Why might equivariance be particularly valuable for medical imaging applications?

## Upload Instructions

Create a `tar.gz` archive called `assignment_e.tar.gz` containing your `cifar10_net` and
`medmnist_equivariant` directories and upload it on Canvas. Only include source code files, i.e.,
exclude cached files, training data, and checkpoints.

```bash
tar -czvf assignment_e.tar.gz cifar10_net medmnist_equivariant
```

Also submit your WandB report as a PDF and provide a link to the online report.
