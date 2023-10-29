import mlrun
from kfp import dsl


@dsl.pipeline(name="Main Pipeline")
def pipeline(
    dataset_name: str,
    dataset_text_field: str,
    urls_file: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    regular_model_name: str,
    pirate_model_name: str,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
):
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest dolly pirate data
    ingest_dolly = project.run_function(
        function="get-pirate-dolly-data",
        params={
            "dataset_name": dataset_name,
        },
        outputs=["regular-dataset", "pirate-dataset"],
    )

    # Ingest MLOps blogs into vector store
    ingest_urls = project.run_function(
        function="ingest-urls", params={"urls_file": urls_file}
    )

    # Configure tuning function
    tuning_fn = project.get_function("fine-tune")
    tuning_fn.with_limits(cpu=6, mem="50Gi", gpus=1)
    tuning_fn.set_env("CUDA_VERSION", "")
    tuning_fn.with_node_selection(
        node_selector={"app.iguazio.com/node-group": "added-v100"}
    )

    # Tune regular model
    tune_regular = project.run_function(
        function=tuning_fn,
        inputs={"dataset": ingest_dolly.outputs["regular-dataset"]},
        params={
            "model_name": regular_model_name,
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "dataset_text_field": dataset_text_field,
            "max_steps": max_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
        },
        outputs=["model"],
    )

    # Tune pirate model
    tune_pirate = project.run_function(
        function=tuning_fn,
        inputs={"dataset": ingest_dolly.outputs["pirate-dataset"]},
        params={
            "model_name": pirate_model_name,
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "dataset_text_field": dataset_text_field,
            "max_steps": max_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
        },
        outputs=["model"],
    )
