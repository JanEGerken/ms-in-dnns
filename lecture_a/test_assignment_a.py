import subprocess
from argparse import Namespace


def test_imports():

    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import pytorch_lightning as pl  # noqa: F401
    from google.cloud import aiplatform, storage  # noqa: F401
    from google.oauth2 import service_account  # noqa: F401

    print("Imports successful!")


def test_sequencer():
    import sequencer

    args = Namespace(sequence="fibonacci", length=10)
    result = sequencer.main(args)
    target = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert result == target, f"got {result}, expected {target}"

    args = Namespace(sequence="prime", length=10)
    result = sequencer.main(args)
    target = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert result == target, f"got {result}, expected {target}"

    args = Namespace(sequence="square", length=10)
    result = sequencer.main(args)
    target = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    assert result == target, f"got {result}, expected {target}"

    args = Namespace(sequence="triangular", length=10)
    result = sequencer.main(args)
    target = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
    assert result == target, f"got {result}, expected {target}"

    args = Namespace(sequence="factorial", length=10)
    result = sequencer.main(args)
    target = [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    assert result == target, f"got {result}, expected {target}"

    subprocess.run("python sequencer.py --sequence fibonacci", shell=True)
    subprocess.run("python sequencer.py --sequence fibonacci --length 10", shell=True)
    subprocess.run("python sequencer.py --sequence prime --length 10", shell=True)
    subprocess.run("python sequencer.py --sequence square --length 10", shell=True)
    subprocess.run("python sequencer.py --sequence triangular --length 10", shell=True)
    subprocess.run("python sequencer.py --sequence factorial --length 10", shell=True)

    output = subprocess.run(
        "python sequencer.py --sequence xxx --length 10", shell=True, stderr=subprocess.PIPE
    )
    assert "invalid choice" in output.stderr.decode()

    print("All sequences have passed the test!")

if __name__ == "__main__":
    test_imports()
    test_sequencer()
