import mlrun
from kfp import dsl


@dsl.pipeline(name="Fine Tuning Pipeline")
def pipeline(
    dataset: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    model_name: str,
    max_steps: int,
    logging_steps: int,
    save_steps: int
):
    # Get our project object
    project = mlrun.get_current_project()

    # Tune model
    tuning_fn = project.get_function("fine-tune")
    tuning_fn.with_limits(cpu=6, mem="50Gi", gpus=1)
    tuning_fn.set_env("CUDA_VERSION", "")
    project.run_function(
        function=tuning_fn,
        inputs={
            "dataset" : dataset
        },
        params={
            "model_name" : model_name,
            "pretrained_tokenizer" : pretrained_tokenizer,
            "pretrained_model" : pretrained_model,
            "dataset_text_field" : "text",
            "max_steps" : max_steps,
            "logging_steps" : logging_steps,
            "save_steps" : save_steps
        },
    )