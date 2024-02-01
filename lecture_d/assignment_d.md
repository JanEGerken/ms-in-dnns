# Assignment D (60 Points)

## Task 1: Extend training script (15 Points)
In Lecture D, we wrote a simple training script for the [Adult income data from the UCI](https://archive.ics.uci.edu/dataset/2/adult). You can find the script in [`income_net.py`](income_net.py). Download the dataset, unpack it into a subdirectory `adult_data` of the root-level `data` directory and verify that the script runs. Although we have implemented training, validation and logging in the lecture, two important pieces of a complete training run are still missing: checkpointing and predicting.

Add a checkpointing mechanism to the `income_net.py` training script. In particular, at the end of each epoch, the model- and optimizer state should be saved to disc. On Google Cloud, the checkpoint should be saved to the same directory in which the output text file is saved. You should only keep the last checkpoint (they can become large for bigger models). Additionally, always keep the best checkpoint according to validation loss. Therefore, you should end up with two checkpoints at the end of training: The best and the last one.

Compute the validation accuracy not only at the end of training but after each epoch and select the best epoch based on this metric instead. Rewrite the final evaluation to use the best epoch (according to validation accuracy), instead of the last one. Discuss potential pitfalls of this strategy.

To get an idea of how the model is behaving, it is very helpful to look at some example predictions. Using the best epoch, predict on ten samples from each class of the validation data and log the results using a [`wandb.Table`](https://docs.wandb.ai/guides/track/log/log-tables#create-tables), as well as to terminal output.

## Task 2: Predicting income (20 Points)
In the lecture, we only looked at training runs with 10 epochs. Train the model for 50 epochs and see if you get a higher final accuracy. What is the reason for the performance you see?

To get a better understanding of what the model is doing, compute the [confusion
matrix](https://en.wikipedia.org/wiki/Confusion_matrix), normalized over targets (i.e. the sums
along rows should be one). Use the function `wandb.plot.confusion_matrix` documented [here](https://docs.wandb.ai/guides/track/log/plots#model-evaluation-charts) to compute and log the confusion matrix. Hint: After one epoch, you should obtain the confusion matrix
```
[[0.9745, 0.0255],
 [0.7589, 0.2411]]
```
Compute the confusion matrix after 10 and after 50 training epochs. Interpret your results.

Using the insights you have gained, improve the training procedure so that the same model reaches a validation performance of about 84% after 200 epochs. Try the following three strategies:
1. Re-weight the classes in the loss, cf. the [`CrossEntropyLoss` documentation](https://pytorch.org/docs/1.13/generated/torch.nn.CrossEntropyLoss.html)
2. Use a learning rate scheduler as documented [here](https://pytorch.org/docs/1.13/optim.html#how-to-adjust-learning-rate) to adjust the learning rate during training. It is sufficient if you try the `StepLR` scheduler.
3. Re-sample the **training** data by repeating the high-income examples until there are equally many high-income and low-income examples. To this end, write a class `ResampledDataset` which takes a dataset as the only argument to its constructor and inherits from `torch.utils.data.Dataset`. The samples should be randomly shuffled. There is a test of for this class in `test_assignment_d.py`.

For all the new training options you add to your model, make the additional hyperparameters
`argarse` arguments with default values corresponding to the previous behaviour. Discuss the results of trying out the different strategies and compare how well they worked.

Summarize your work in a [Weights and Biases Report](https://docs.wandb.ai/guides/reports). Create
a report, publish it to your project and then save it as a PDF. These reports also allow you to add
text/markdown fields, use them for your discussion. Submit the PDF on Canvas and add the URL to your
online report as well.

## Task 3: CIFAR10 and data augmentation (25 Points)

In this exercise, we start exploring the power of deep learning by training a CNN on the
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset. This dataset
consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000
training images and 10000 test images. The classes are `["airplane", "automobile", "bird", "cat",
"deer", "dog", "frog", "horse", "ship", "truck"]`. Create a `cifar_net.py` file for which you can adapt
the `incomet_net.py` script you wrote in the previous exercise to load the CIFAR10 data using the dataset provided by `torchvision`. To use the images as input to the model, you need to transform them to torch tensors and adjust their range. You can do this using `torchvision.transforms`
```python
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
```
The model we are going to use in this exercise is a variant of the popular VGG16 architecture, introduced for the ImageNet dataset [here](https://arxiv.org/abs/1409.1556). In PyTorch, we can implement this model as
```python
class CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
This model has about 40M trainable parameters and can only be reasonably trained on a GPU. Replace the prediction logging from the `income_net.py` script by a new logging mechanism which populates a `wandb.Table` with columns image, ground truth and prediction. For 5 validation samples from each class, log the image, as well as the names of the ground truth and predicted classes. You can have a look at  [this](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb) notebook for inspiration. In the web UI, group the table by the ground-truth column to get a convenient overview. Make sure to log your runs to a new wandb project. For debugging the data generation and logging, it is advisable to use a small dummy-model which can be trained locally on the CPU. Once logging and data loading work as intended, you can train the full model on a Google Cloud GPU. When training the full model, be careful with the size of the checkpoints which will be several hundred MB.

Train the model specified above with batch size 32 for a few epochs and observe the loss curves.

Next, add an `argparse` option called `--batchnorm` to add a `torch.nn.BatchNorm2d` layer after each convolutional layer. Train the model for 60 epochs and interpret the training- and validation loss curves. This run should take about 1h to complete.

Next, add an option `--dropout` for a `torch.nn.Dropout` layer with dropout probability `0.3` after each pooling layer and after the ReLU layers in the fully connected classifier. Train this model with batch norm and dropout for 60 epochs and interpret the results.

Finally, add an option `--augment` for data augmentation. Augment the **training** data using `torchvision.transforms` with random horizontal- and vertical flips, random rotations by -10 to 10 degrees, random resized crops with a scale of 0.8 to 1.0 and an aspect ratio of 0.8 to 1.2 and a color jittering with brightness factor, contrast factor, saturation factor and hue factor in $[0.8, 1.2]$. Train again for 60 epochs and interpret the results. Also try combining data augmentation and dropout.

Summarize your results in a WandB report, including the table of predicted results and the learning curves.

