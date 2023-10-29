import os
import shutil
import tempfile
from abc import ABC
from typing import Dict, List

import mlrun
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
import mlrun
from mlrun.artifacts.manager import Artifact, PlotlyArtifact
from mlrun.execution import MLClientCtx
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import create_class
from peft import LoraConfig
from plotly import graph_objects as go
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ["HUGGING_FACE_HUB_TOKEN"] = mlrun.get_secret_or_env("HUGGING_FACE_HUB_TOKEN")


# ----------------------from MLRUN--------------------------------
class HFTrainerMLRunInterface(MLRunInterface, ABC):
    """
    This is temporary and will be built in mlrun 1.5.0
    Interface for adding MLRun features for tensorflow keras API.
    """

    # MLRuns context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-huggingface"

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = [
        "train",
        # "evaluate"
    ]

    @classmethod
    def add_interface(
        cls,
        obj: Trainer,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        super(HFTrainerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_train(cls):
        def wrapper(self: Trainer, *args, **kwargs):
            # Restore the evaluation method as `train` will use it:
            # cls._restore_attribute(obj=self, attribute_name="evaluate")

            # Call the original fit method:
            result = self.original_train(*args, **kwargs)

            # Replace the evaluation method again:
            # cls._replace_function(obj=self, function_name="evaluate")

            return result

        return wrapper


class MLRunCallback(TrainerCallback):
    """
    This is temporary and will be built in mlrun 1.5.0
    Callback for collecting logs during training / evaluation of the `Trainer` API.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model_name: str = "model",
        tag: str = "",
        labels: Dict[str, str] = None,
        extra_data: dict = None,
    ):
        super().__init__()

        # Store the configurations:
        self._context = (
            context
            if context is not None
            else mlrun.get_or_create_ctx("./mlrun-huggingface")
        )
        self._model_name = model_name
        self._tag = tag
        self._labels = labels
        self._extra_data = extra_data if extra_data is not None else {}

        # Set up the logging mode:
        self._is_training = False
        self._steps: List[List[int]] = []
        self._metric_scores: Dict[str, List[float]] = {}
        self._artifacts: Dict[str, Artifact] = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._steps.append([])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        recent_logs = state.log_history[-1].copy()

        recent_logs.pop("epoch")
        current_step = int(recent_logs.pop("step"))
        if current_step not in self._steps[-1]:
            self._steps[-1].append(current_step)

        for metric_name, metric_score in recent_logs.items():
            if metric_name.startswith("train_"):
                if metric_name.split("train_")[1] not in self._metric_scores:
                    self._metric_scores[metric_name] = [metric_score]
                continue
            if metric_name not in self._metric_scores:
                self._metric_scores[metric_name] = []
            self._metric_scores[metric_name].append(metric_score)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._is_training = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

        if self._is_training:
            return

    def _log_metrics(self):
        for metric_name, metric_scores in self._metric_scores.items():
            self._context.log_result(key=metric_name, value=metric_scores[-1])
            if len(metric_scores) > 1:
                self._log_metric_plot(name=metric_name, scores=metric_scores)
        self._context.commit(completed=False)

    def _log_metric_plot(self, name: str, scores: List[float]):
        # Initialize a plotly figure:
        metric_figure = go.Figure()

        # Add titles:
        metric_figure.update_layout(
            title=name.capitalize().replace("_", " "),
            xaxis_title="Samples",
            yaxis_title="Scores",
        )

        # Draw:
        metric_figure.add_trace(
            go.Scatter(x=np.arange(len(scores)), y=scores, mode="lines")
        )

        # Create the plotly artifact:
        artifact_name = f"{name}_plot"
        artifact = PlotlyArtifact(key=artifact_name, figure=metric_figure)
        self._artifacts[artifact_name] = self._context.log_artifact(artifact)


def apply_mlrun(
    trainer: transformers.Trainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    """
    This is temporary and will be built in mlrun 1.5.0
    """
    # Get parameters defaults:
    if context is None:
        context = mlrun.get_or_create_ctx(HFTrainerMLRunInterface.DEFAULT_CONTEXT_NAME)

    HFTrainerMLRunInterface.add_interface(obj=trainer)

    if auto_log:
        trainer.add_callback(
            MLRunCallback(
                context=context,
                model_name=model_name,
                tag=tag,
                labels=labels,
                extra_data=extra_data,
            )
        )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train(
    context: MLClientCtx,
    model_name: str,
    dataset: pd.DataFrame,
    pretrained_tokenizer: str,
    pretrained_model: str,
    dataset_text_field: str,
    model_class: str = "transformers.AutoModelForCausalLM",
    tokenizer_class: str = "transformers.AutoTokenizer",
    max_steps: int = 100,
    save_steps: int = 50,
    logging_steps: int = 10,
):
    torch.cuda.empty_cache()

    # Prepare tokenizer and dataset
    tokenizer_class = create_class(tokenizer_class)

    tokenizer = tokenizer_class.from_pretrained(
        pretrained_tokenizer,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_pandas(df=dataset)

    # Configure quanitization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = create_class(model_class).from_pretrained(
        pretrained_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    # Prepare training arguments
    training_arguments = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=max_steps,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        ddp_find_unused_parameters=False,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "k_proj",
            "v_proj",
        ],  # Choose all linear layers from the model
    )

    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=dataset_text_field,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    apply_mlrun(trainer, model_name=model_name)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # Apply training with evaluation:
    context.logger.info(f"training '{model_name}'")
    trainer.train()

    temp_directory = tempfile.TemporaryDirectory().name
    trainer.save_model(temp_directory)

    # Zip the model directory:
    shutil.make_archive(
        base_name="model",
        format="zip",
        root_dir=temp_directory,
    )

    # Log the model:
    context.log_model(
        key="model",
        db_key=model_name,
        model_file="model.zip",
        tag="",
        framework="Hugging Face",
    )
