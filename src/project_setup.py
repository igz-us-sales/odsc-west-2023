import os

import mlrun


def create_and_set_project(
    name: str,
    source: str,
    cpu_image: str = "nschenone/odsc-west-2023-cpu:1.4.1",
    gpu_image: str = "nschenone/odsc-west-2023-gpu:1.4.1",
    artifact_path: str = None,
    user_project: bool = False,
    secrets_file: str = None,
    force_build: bool = False,
):
    # Set environment secrets via secrets file
    if secrets_file and os.path.exists(secrets_file):
        mlrun.set_env_from_file(secrets_file)

    # Get / Create a project from the MLRun DB:
    project = mlrun.get_or_create_project(
        name=name, context="./", user_project=user_project
    )

    # Set MLRun project secrets via secrets file
    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)

    # Set artifact path
    if artifact_path:
        project.artifact_path = artifact_path

    # Load artifacts
    project.register_artifacts()

    # Export project to zip if relevant
    if ".zip" in source:
        print(f"Exporting project as zip archive to {source}...")
        project.export(source)

    # Set the project source
    project.set_source(source, pull_at_runtime=True)

    # Set MLRun functions
    project.set_function(
        name="get-pirate-dolly-data",
        func="src/functions/get_pirate_dolly_data.py",
        kind="job",
        handler="get_pirate_dolly_data",
        image=cpu_image,
    )

    project.set_function(
        name="ingest-urls",
        func="src/functions/get_blog_data.py",
        kind="job",
        handler="ingest_urls",
        image=cpu_image,
        with_repo=True,
    )

    project.set_function(
        name="fine-tune",
        func="src/functions/fine_tune.py",
        kind="job",
        handler="train",
        image=gpu_image,
    )

    serving_fn = project.set_function(
        "src/functions/serving.py",
        name="serving",
        kind="serving",
        image=gpu_image,
        with_repo=True,
    )

    # Set MLRun workflows
    project.set_workflow(
        name="ingest", workflow_path="src/workflows/ingest_workflow.py"
    )
    project.set_workflow(name="tune", workflow_path="src/workflows/tune_workflow.py")
    project.set_workflow(name="main", workflow_path="src/workflows/main_workflow.py")

    # Save and return the project:
    project.save()
    return project
