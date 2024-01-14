# Lecture A: Python and the compute environment
In this lecture, we cover the basic Python setup for the compute environment and how to program effectively.

## Learning Python
A basic understanding of Python is a prerequisite for this course. If you want to refresh your knowledge, have a look at
- [learnpython.org](https://www.learnpython.org/) for some basics
- [Real Python](https://realpython.com/) for in-depth articles on basic and advanced topics

## Git
You will be using the version control system [git](https://git-scm.com/) during this course. Install git as explained [here](https://github.com/git-guides/install-git) if you do not have it already. For a good introduction to git, have a look at [Atalassin Git](https://www.atlassian.com/git).

## GitHub and the course repository
The code for the assignments and instructions for implementations will be provided in this GitHub repository. To start working with it
1. Create a GitHub account if you do not have one
2. It is very convenient and safer to set up SSH Keys to interact with GitHub from your local git client. Follow the steps starting [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys) to do this
3. Fork and clone this repository as explained [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). You do not need to add the upstream remote on your local repo if you do not want to, you can use the GitHub website to sync with the upstream repository as explained in the first section [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

When the course repo is updated, you should see a banner in your forked repository that your repo is "XX commits behind". If you have done changes, you repo is "XX commits ahead". Make sure to sync your repo when your repo is behind. If you have commits you think are useful for everyone (e.g. fixing typos) you can open a pull request.

## IDEs (integrated development environments)
In order to work effectively with Python, it is recommended that you install an IDE (a text editor which helps you with syntax highlighting and -checking, formatting etc). Popular choices are e.g.
- [VisualStudioCode](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)

VisualStudioCode is very extendible. For us, the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens) are the most useful.

For easily keeping well-formatted code, you can use a code formatter, like [black](https://black.readthedocs.io/en/stable/index.html). You can configure your IDE to run black on any Python file upon saving, as explained [here](https://code.visualstudio.com/docs/python/formatting) for VSCode.

A linter is a program that helps you spot syntax errors immediately while writing code and warns you about potential errors. A popular Python linter is [flake8](https://flake8.pycqa.org/en/latest/). Linters can integrate into your IDE to give you immediate visual feedback. Instructions for how to enable this in VSCode can be found [here](https://code.visualstudio.com/docs/python/linting).

This repository contains config files `pyproject.toml` for black and `.flake8` for flake8 which set them to a maximum line length of 100 characters and make them work well together.

## Python import system
A good article with many details about the Python import system can be found [here](https://realpython.com/python-import/).

## Setting up the local environment
For working with the code in this repository and solve the assignments, it is recommended that you create a Python virtual environment:
1. Create a Python virtual environment in the cloned repository and activate it as explained [here](https://docs.python.org/3/library/venv.html)
2. Inside the venv, install the dependencies by running (this will download ~1GB of data)
```
pip3 install --upgrade google-cloud-aiplatform==1.38.1 torch==1.13 lightning==2.1.2 torchvision==0.14.0 matplotlib==3.8.2 pandas==2.1.4 wandb==0.16.1 jsonargparse[signatures]==4.27.1 rich==13.7.0
```

These package versions should work with Python 3.10. You can check your Python version by running `python3 --version`.

## Debugging
Debugging is the process of finding and fixing errors in programs. Debuggers are programs that help in that by letting the user stop the execution of a program at a certain line of code (breakpoint) and then step through the code line-by-line. Python comes with the debugger [PDB](https://docs.python.org/3/library/pdb.html) which is simple, yet powerful. For a tutorial on how to use PDB, see e.g. [here](https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger). IDEs provide a visual layer for the debugger. For debugging in VSCode, see [this](https://code.visualstudio.com/docs/editor/debugging) page.
