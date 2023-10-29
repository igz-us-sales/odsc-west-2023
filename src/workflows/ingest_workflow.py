import mlrun
from kfp import dsl


@dsl.pipeline(name="Data Ingestion Pipeline")
def pipeline(dataset_name: str):
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest dolly pirate data
    ingest = project.run_function(
        function="get-pirate-dolly-data",
        params={
            "dataset_name": dataset_name,
        }
    )