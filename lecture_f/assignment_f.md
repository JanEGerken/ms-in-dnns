# Assignment F: Project proposal and normalizing flows

## Project proposal (mandatory for admission to project)

Write a project proposal of about 2-3 pages length in which you describe what you plan to do. Be as specific as possible. Your proposal should include the following sections:

### Abstract
A short summary (about 1/2 page) of the project idea. You should answer the questions
- What is the context of the project (e.g. equivariant NN for medical images)?
- What do you want to do?
- What do you hope to achieve?

You can check some ML papers to get an idea of what an abstract typically looks like. You can look e.g. [here](http://arxiv.org/abs/1412.6572), [here](https://arxiv.org/abs/1801.10130) or [here](http://arxiv.org/abs/2307.07313).

### Dataset
To do any deep learning project which involves experiments, you need data. Therefore, thinking carefully about what data to use is critical to a good project. In this section you should answer
- What data do you want to use, a publicly available dataset or self-generated data?
  - If you want to download a dataset, provide a link to the dataset or cite a reference paper.
  - If you want to generate the data yourself, think about possible difficulties.
- How big is the dataset, what format do the samples have (e.g. image size, file format etc.)?
- Do you expect that you need to do pre-processing of the data?

### Model
If you plan to use a complex model, it is a good idea to use a pre-existing implementation. In this case, you should verify that the model was written a reasonably recent version of PyTorch. Furthermore, a lot of time can be spent tuning hyperparameters, in particular for large models with many parameters. Therefore, it is a good idea to look in the literature for some values that you could try first.
This section should answer
- Do you want to train one model or several?
- What model/models do you want to train on the dataset?
- How big do you expect this model to be, is it feasible to train on one GPU?
- Are there implementations in PyTorch available for this model that you can use or do you plan to implement the model from scratch?
- What hyperparameters do you expect will be important to tune? Are there reasonable starting values available in the literature?
- What loss do you intend to use / which losses do you want to try?

### Evaluation
When tuning hyperparameters or comparing different approaches, it is important to have **one** (scalar) metric to measure performance. This should be implemented early on in the project and be tested so that you can rely on it. Further metrics can then be used to get a more differentiated picture of the performance of your most important models. This section should answer
- How do you want to evaluate your results?
- What will be the metric you will use for hyperparameter tuning and model comparison?
- What other metrics will you use?

### Potential problems and backup plan
Research project never work out as planned - this lies in the nature of research. Therefore, it is good to be aware of potential obstacles and have contingency plans. Think about what issues might arise and how you might overcome them, e.g. you could use a different dataset, a smaller model or restrict your comparison to fewer cases. Also, think about a "minimally working product", i.e. a part of the project that you are sure will work and you could write a report about, even though it might not be that exciting. This section should answer
- What do you think will be the biggest challenge in realizing this project? What would be ways to overcome it?
- What part of the project are you quite certain will work that you could fall back on?

## Normalizing flows (25 Points)

Use the provided package `mnist_flow` to train a normalizing flow on MNIST as discussed in the lecture. Training for 200 epochs will take about 4:15h on Google Cloud. Both of the following tasks can be performed on the CPU. Hence, download the best checkpoint from your training run from the Google Cloud Storage Bucket and load the checkpoint of the trained model for the following tasks.


### Interpolation (15 Points)

For a well-trained normalizing flow is distribution of samples in the base space approximately Gaussian. Due to the spherical symmetry of the normal distribution, this means that performing a linear interpolation between images in the base space will likely only traverse regions of high probablility, i.e. all samples along the line connecting two samples in the base space will approximate realistic samples from the data distribution. We want to try this out in this exercise.

Rewrite the `mnist_flow` package such that
- you can initialize a model by loading a checkpoint from disk.
- all base space dimensions are returned by the `reverse=False` case of the `forward` call. Remember that the `SplitFlow` model as is only returns the dimensions which are to be transformed further. The final return value should contain as many dimensions as the input. Similarly, the `forward` method should also accept the same number of input dimensions in the `reverse=True` case. In the end, the flow should be completely reversible in this way.

Perform linear interpolation between validation samples of two (different) classes in pixel space and in the base space. In the latter case, map the samples back into pixel space. Show 10 images from the interpolation, where the first corresponds to the initial sample and the last to the final sample in a different class.

Plot both interpolations in a two-row grid using `torchvision.utils.make_grid` ([documentation](https://pytorch.org/vision/0.14/generated/torchvision.utils.make_grid.html)) and log the result as an `wandb.Image`.

Try different initial and final classes and discuss the differences you observe between pixel-space and base-space interpolations in a WandB report.


### Anomaly detection (10 Points)

By comparing the negative log-likelihood of samples, one can perform anomaly detection using normalizing flows: If the NLL of a sample lies above a threshold is it deemed unlikely in the learned probability distribution and hence discarded as an anomaly (an out of distribution sample).

As a source of anomalies, use the [Kuzushiji-MINST dataset](https://pytorch.org/vision/0.14/generated/torchvision.datasets.KMNIST.html), a drop-in replacement for MNIST showing handwritten Japanese characters. Add KMNIST to the `DataModule` and log a grid with a batch of images of MNIST and KMNIST to WandB to get an idea of what the samples look like. You can use the `torchvision.utils.make_grid` function again.

Then, go through the validation data of MNIST and KMNIST and perform anomaly detection as outlined above. Log the results as a binary confusion matrix in WandB. You can re-use the code you wrote to log the confusion matrix in Assignment E. Try different thresholds and observe how the results change. Try to find a good threshold which minimizes confusion.

Finally, log ten false positives (samples from KMNIST which are predicted to lie in MNIST) and log a grid of them to WandB to get an idea of what difficult samples look like.

Summarize your results in a WandB report.
