# Assignment D (60 Points)

## Task 1: Extend training script (15 Points)
In Lecture D, we wrote a simple training script for the [Adult income data from the UCI](https://archive.ics.uci.edu/dataset/2/adult). You can find the script in [`income_net.py`](income_net.py). Download the dataset, unpack it into a subdirectory `adult_data` of the root-level `data` directory and verify that the script runs. Although we have implemented training, validation and logging in the lecture, two important pieces of a complete training run are still missing: checkpointing and predicting.

Add a checkpointing mechanism to the `income_net.py` training script. In particular, at the end of each epoch, the model- and optimizer state should be saved to disc. On Google Cloud, the checkpoint should be saved to the same directory in which the output text file is saved. You should only keep the last checkpoint (they can become large for bigger models). Additionally, always keep the best checkpoint according to validation loss. Therefore, you should end up with two checkpoints at the end of training: The best and the last one.

Compute the validation accuracy not only at the end of training but after each epoch, log it and select the best epoch based on this metric instead. Rewrite the final evaluation to use the best epoch (according to validation accuracy), instead of the last one. Discuss potential pitfalls of this strategy.

To get an idea of how the model is behaving, it is very helpful to look at some example predictions. Using the best epoch, predict on ten samples from each class of the validation data and log the results using a [`wandb.Table`](https://docs.wandb.ai/guides/track/log/log-tables#create-tables), as well as to terminal output.

## Task 2: Predicting income (20 Points)
In the lecture, we only looked at training runs with 10 epochs. Train the model for 50 epochs and see if you get a higher final accuracy. What is the reason for the performance you see?

To get a better understanding of what the model is doing, compute the [confusion
matrix](https://en.wikipedia.org/wiki/Confusion_matrix), normalized over targets (i.e. the sums
along rows should be one). Use the function `wandb.plot.confusion_matrix` documented
[here](https://docs.wandb.ai/guides/track/log/plots#model-evaluation-charts) to compute and log the
confusion matrix. Hint: After one epoch, you should obtain a confusion matrix similar to
```
[[0.9745, 0.0255],
 [0.7589, 0.2411]]
```
(the exact values depend on various factors as can be read
[here](https://pytorch.org/docs/stable/notes/randomness.html)).
Compute the confusion matrix after 10 and after 50 training epochs. Interpret your results.

Using the insights you have gained, improve the training procedure so that the same model reaches a validation performance of about 84% after 200 epochs. Try the following three strategies:
1. Re-weight the classes in the loss, cf. the [`CrossEntropyLoss` documentation](https://pytorch.org/docs/1.13/generated/torch.nn.CrossEntropyLoss.html)
2. Use a learning rate scheduler as documented [here](https://pytorch.org/docs/1.13/optim.html#how-to-adjust-learning-rate) to adjust the learning rate during training. It is sufficient if you try the `StepLR` scheduler.
3. Re-sample the **training** data by repeating the high-income examples until there are equally many high-income and low-income examples. To this end, write a class `ResampledDataset` which takes a dataset as the only argument to its constructor and inherits from `torch.utils.data.Dataset`. The samples should be randomly shuffled. There is a test of for this class in `test_assignment_d.py`.

For all the new training options you add to your model, make the additional hyperparameters
`argparse` arguments with default values corresponding to the previous behaviour. Discuss the results of trying out the different strategies and compare how well they worked.

Summarize your work in a [Weights and Biases Report](https://docs.wandb.ai/guides/reports). Create
a report, publish it to your project and then save it as a PDF. These reports also allow you to add
text/markdown fields, use them for your discussion. Submit the PDF on Canvas and add the URL to your
online report as well.

## Task 3: Adversarial Attacks (25 Points)

As discussed in Lecture 4, it is not hard to find images which fool a well-trained classifier into
giving the wrong prediction, even with no perceptible difference to a correctly classified image.
This is known as adversarial vulnerability. In this exercise, you will compute adversarial examples
for a pre-trained CIFAR-10 classifier.

A pre-trained `SimpleCIFARNet` model is provided in the [`cifar10_simple`](cifar10_simple/) package
along with a checkpoint file `pretrained_cifar10.ckpt`. Install the package in editable mode:

```bash
cd cifar10_simple
pip install -e .
```

Write a script `adv_attacks.py` (you can use [`adv_attacks_template.py`](adv_attacks_template.py) as
a starting point) which logs to a new WandB project `ms-in-dnns-cifar10-adv-attacks`. For data
loading, you can use the functions provided in the `cifar10_simple` package and load the pre-trained
model using the `load_pretrained` function.

Your script should have the following `argparse` arguments:
- `--data-root` for the directory where to find the CIFAR10 data
- `--run-name` for the name of the run
- `--ckpt-path` for the path to the checkpoint to be used
- `--source-class` the class of the samples to be optimized (one of the 10 CIFAR-10 class names)
- `--n-samples` the number of samples to start the attack from (default: `5`)
- `--max-iter` the maximum number of optimization iterations (default: `1000`)
- `--lr` learning rate of the optimizer (default: `1e-3`)
- `--prob-threshold` the predicted probability of the target class at which to stop the optimization (default: `0.99`)

First, find `--n-samples` samples in the validation data which lie in the source class. Then, for
each of the 10 classes as target class, optimize the input to the network using the Adam optimizer
(with `maximize=True`) until the maximum number of iterations is reached or the predicted
probability for the target class surpasses the probability threshold. It is easiest to do this in
pure PyTorch and not Lightning. You should end up with `n_samples * 10` new images.

Log your results into a WandB table with columns:
- `source_image`: the original image
- `gt_class`: ground truth class (the source class)
- `target_class`: the class the attack targets
- `adversary`: the modified image
- `diff`: rescaled pixel-by-pixel difference to original (choose a scaling that makes the difference visible)
- `target_prob`: final target probability

Pick a source class with high accuracy (e.g., "ship" or "automobile" typically work well) and run
the attack with the defaults given above. Discuss your results in a WandB report: How successful
are the attacks? How perceptible are the perturbations? Are some target classes harder to achieve
than others?

Generate the report as a PDF and provide also a link to the online report in your submission.

