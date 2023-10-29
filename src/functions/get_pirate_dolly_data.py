import copy

import mlrun
import pandas as pd
from arrr import translate
from datasets import Dataset, load_dataset

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)


def apply_prompt_template(examples):
    instruction = examples["instruction"]
    response = examples["response"]
    context = examples.get("context")

    if context:
        full_prompt = PROMPT_WITH_INPUT_FORMAT.format(
            instruction=instruction, response=response, input=context
        )
    else:
        full_prompt = PROMPT_NO_INPUT_FORMAT.format(
            instruction=instruction, response=response
        )
    return {"text": full_prompt}


@mlrun.handler(outputs=["regular-dataset", "pirate-dataset"])
def get_pirate_dolly_data(
    dataset_name: str,
) -> pd.DataFrame:
    # Load dolly data
    regular_dataset = load_dataset(dataset_name, split="train")

    # Apply pirate formatting to response
    pirate_dataset = copy.deepcopy(regular_dataset)
    df = pirate_dataset.to_pandas()
    df["response"] = df["response"].apply(translate)
    pirate_dataset = Dataset.from_pandas(df=df)

    # Apply prompt template
    regular_dataset = regular_dataset.map(apply_prompt_template)
    pirate_dataset = pirate_dataset.map(apply_prompt_template)

    return regular_dataset.to_pandas(), pirate_dataset.to_pandas()
