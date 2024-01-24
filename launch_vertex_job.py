import os
import argparse
import subprocess
from datetime import datetime
from pathlib import PurePath, PurePosixPath
import json

from google.cloud import aiplatform, storage
from google.oauth2 import service_account

PROJECT = "msdnn-lectures"
REGION = "europe-west4"
BUCKET = "gs://msindnn_staging"
EXPERIMENT = "msdnn-assignments"
EXP_DESCRIPTION = "Assignments for MS in DNNs lecture"
WANDB_KEY = json.load(open("wandb_key.json"))
CREDENTIALS = service_account.Credentials.from_service_account_file("credentials.json")
CONTAINER = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
N_GPUS = 1
DATETIME_FMT = "%Y-%m-%d_%H%M%S"

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project=PROJECT,
    # the Vertex AI region you will use
    # defaults to us-central1
    location=REGION,
    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket=BUCKET,
    # custom google.auth.credentials.Credentials
    # environment default credentials used if not set
    credentials=CREDENTIALS,
    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment=EXPERIMENT,
    # description of the experiment above
    experiment_description=EXP_DESCRIPTION,
)


def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client(project=PROJECT, credentials=CREDENTIALS)
    bucket = storage_client.bucket(BUCKET.split("gs://")[1])
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    destination_file_name = os.path.join(BUCKET, destination_blob_name)

    return destination_file_name


def launch_script_job(args):
    timestamp = datetime.now().strftime(DATETIME_FMT)
    log_path = PurePosixPath(BUCKET.replace("gs://", "/gcs/"), f"{args.name}_{timestamp}", "log.txt")
    requirements = [
        "torch==1.13",
        "lightning==2.1.2",
        "torchvision==0.14.0",
        "matplotlib==3.8.2",
        "pandas==2.1.4",
        "wandb==0.16.1",
        "jsonargparse[signatures]==4.27.1",
        "rich==13.7.0",
    ]

    job = aiplatform.CustomJob.from_local_script(
        credentials=CREDENTIALS,
        display_name=args.name,
        script_path=args.path,
        environment_variables={
            "LOG_PATH": str(log_path),
            "CREATION_TIMESTAMP": timestamp,
            "WANDB_KEY": WANDB_KEY,
        },
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=N_GPUS,
        replica_count=1,
        args=args.args,
        container_uri=CONTAINER,
        requirements=requirements + args.requirements,
        labels={},
    )

    job.run()
    return job


def launch_package_job(args):
    timestamp = datetime.now().strftime(DATETIME_FMT)
    log_path = PurePosixPath(
        BUCKET.replace("gs://", "/gcs/"),
        "custom-training-python-package",
        args.name,
        timestamp,
        "log.txt",
    )

    print("Creating source distribuion")
    subprocess.run(
        [
            "python3",
            "setup.py",
            "sdist",
            "--formats=gztar",
            "--dist-dir=" + str(PurePath("dist", timestamp)),
        ],
        cwd=args.directory,
    )

    dist_dir = PurePath(args.directory, "dist", timestamp)
    candidates = []
    for entry in dist_dir.iterdir():
        if ".tar.gz" in entry.name:
            candidates.append(entry)
    assert len(candidates) == 1, f"found more than one .tar.gz-file in {dist_dir}: {candidates}"
    source_name = candidates[0].name
    destination_blob_name = PurePosixPath("custom-training-python-package", f"{args.name}",
                                          f"{timestamp}/{source_name}")
    python_package_gcs_uri = upload_blob(str(candidates[0]), str(destination_blob_name))
    python_module_name = args.task_module

    print(f"Custom Training Python Package is uploaded to: {python_package_gcs_uri}")

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=args.name,
        credentials=CREDENTIALS,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_module_name,
        container_uri=CONTAINER,
    )
    job.run(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=N_GPUS,
        replica_count=1,
        args=args.args,
        environment_variables={
            "LOG_PATH": str(log_path),
            "CREATION_TIMESTAMP": timestamp,
            "WANDB_KEY": WANDB_KEY,
        },
    )
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()

    script_parser = sp.add_parser(
        "script", help="Submit single python script, not bundled as a package"
    )
    script_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the training job",
    )
    script_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the script to be executed",
    )
    script_parser.add_argument(
        "--requirements",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="List of required packages for the script to run, in pip syntax",
    )
    script_parser.add_argument(
        "--args",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Arguments passed to the script",
    )
    script_parser.set_defaults(func=launch_script_job)

    package_parser = sp.add_parser("package", help="Submit bundled python package")
    package_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the training job",
    )
    package_parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Root directory of the package containing the setup.py",
    )
    package_parser.add_argument(
        "--task-module",
        type=str,
        required=True,
        help="Module to be executed in Python import syntax",
    )
    package_parser.add_argument(
        "--args",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Arguments passed to the executed module",
    )
    package_parser.set_defaults(func=launch_package_job)

    args, unknown_args = parser.parse_known_args()
    args.args = unknown_args
    args.func(args)
