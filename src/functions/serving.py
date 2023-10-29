import json
import os
import zipfile
from typing import Any, Dict, Tuple, List

import mlrun
import mlrun.artifacts
import numpy as np
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from mlrun.serving.v2_serving import V2ModelServer
from mlrun.utils import create_class
from peft import PeftModel, PeftConfig
from langchain.docstore.document import Document

from src.config import AppConfig, get_vector_store

config = AppConfig()
store = get_vector_store(config)

os.environ["HUGGING_FACE_HUB_TOKEN"] = mlrun.get_secret_or_env("HUGGING_FACE_HUB_TOKEN")

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}""".format(
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

{response_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)


def parse_documents(relevant_documents: List[Document]) -> Tuple[str, set]:
    context = list()
    sources = set()
    for doc in relevant_documents:
        context.append(doc.page_content)
        sources.add(doc.metadata["source"])

    context = "\n\n".join(context)
    return context, sources

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list):
        StoppingCriteria.__init__(self)
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # print(input_ids)
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def preprocess(request: dict) -> dict:
    """
    convert the request to the required structure for the predict function

    :param request: A http request that contains the prompt
    """
    
    # Read bytes:
    if isinstance(request, bytes):
        request = json.loads(request)
        
    # Get the prompt:
    formatted_prompt = None
    user_prompt = request.pop("prompt")
        
    rag = request.pop("rag")
    k = request.pop("k")
    sources = ""
    
    if rag:
        docs = store.similarity_search(user_prompt, k=k)
        context, sources = parse_documents(docs)
        formatted_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=user_prompt, input=context)
    else:
        formatted_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=user_prompt)

    # Update the request and return:
    request = {"inputs": [{"prompt": [formatted_prompt], "sources" : sources, **request}]}
    return request


class LLMModelServer(V2ModelServer):
    """
    This is temporary and will be built in mlrun 1.5.0
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_class: str = "transformers.AutoModelForCausalLM",
        tokenizer_class: str = "transformers.AutoTokenizer",
        # model args:
        model_args: dict = None,
        # Load from MLRun args:
        model_path: str = None,
        # Load from hub args:
        model_name: str = None,
        tokenizer_name: str = None,
        # peft model:
        adapters: dict = None,
        # Inference args:
        stop_token: str = None,
        **class_args,
    ):
        # Initialize the base server:
        super(LLMModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )

        # Save class names:
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class

        # Save hub loading parameters:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or self.model_name

        # Save load model arguments:
        self.model_args = model_args
        
        # Generation arguments
        self.stop_token = stop_token

        # PEFT adapters:
        self.adapters = adapters

        # Prepare variables for future use:
        self.model = None
        self.tokenizer = None
        self._model_class = None
        self._tokenizer_class = None
        self.stopping_criteria = None

    def load(self):
        # Get classes:
        self._model_class = create_class(self.model_class)
        self._tokenizer_class = create_class(self.tokenizer_class)

        # Load the model and tokenizer:
        if self.model_path:
            self._load_from_mlrun()
        else:
            self._load_from_hub()

        if self.adapters:
            self._load_adapters()
            
        if self.stop_token:
            self._load_stop_criteria()

    def _extract_model(self, url):
        # Get the model artifact and file:
        (
            model_file,
            model_artifact,
            extra_data,
        ) = mlrun.artifacts.get_model(url)

        # Read the name:
        model_name = model_artifact.spec.db_key

        # Extract logged model files:
        model_directory = os.path.join(os.path.dirname(model_file), model_name)
        with zipfile.ZipFile(model_file, "r") as zip_file:
            zip_file.extractall(model_directory)
        return model_directory

    def _load_adapters(self):
        for i, (adapter_name, adapter_path) in enumerate(list(self.adapters.items())):
            # There is different syntax to add the first adapter vs additional
            # See https://github.com/huggingface/peft/issues/957#issuecomment-1733375946
            if i == 0:
                print(f"Loading model with {adapter_name} adapter")
                model_directory = self._extract_model(adapter_path)
                self.model = PeftModel.from_pretrained(model=self.model, adapter_name=adapter_name, model_id=model_directory)
            else:
                print(f"Loading {adapter_name} adapter")
                model_directory = self._extract_model(adapter_path)
                peft_config = PeftConfig.from_pretrained(model_directory)
                self.model.add_adapter(adapter_name=adapter_name, peft_config=peft_config)       
        self.model.eval()

    def _load_from_mlrun(self):
        model_directory = self._extract_model(self.model_path)

        # Loading the saved pretrained tokenizer and model:
        self.tokenizer = self._tokenizer_class.from_pretrained(model_directory)
        self.model = self._model_class.from_pretrained(
            model_directory, torch_dtype=torch.float16, **self.model_args
        )

    def _load_from_hub(self):
        # Loading the pretrained tokenizer and model:
        self.tokenizer = self._tokenizer_class.from_pretrained(
            self.tokenizer_name,
            model_max_length=512,
        )
        self.model = self._model_class.from_pretrained(
            self.model_name, **self.model_args
        )
        
    def _load_stop_criteria(self):
        # Stop model generation when encountering these tokens (increases performance)
        stop_token_ids = self.tokenizer.convert_tokens_to_ids([self.stop_token])
        self.stopping_criteria = StoppingCriteriaList([
            StopOnTokens(stop_token_ids=stop_token_ids)
        ])
        

    def predict(self, request: Dict[str, Any]) -> dict:
        # Get the inputs:
        kwargs = request["inputs"][0]
        adapter = kwargs.pop("adapter")
        prompt = kwargs.pop("prompt")[0]
        sources = kwargs.pop("sources")

        # Tokenize:
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if self.model.device.type == "cuda":
            inputs = inputs.cuda()

        # Get the pad token id:
        pad_token_id = self.tokenizer.eos_token_id

        # Infer through the model:
        self.model.set_adapter(adapter)
        output = self.model.generate(
            input_ids=inputs,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            stopping_criteria=self.stopping_criteria,
            **kwargs,
        )
        print("SOMETHING CHANGED")

        # Detokenize:
        prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove prompt at beginning
        prediction = prediction[len(prompt):]
        
        # Remove garbage at end
        if self.stop_token:
            prediction = prediction.split(self.stop_token)[0].strip()

        return {"prediction": prediction, "prompt": prompt, "sources" : list(sources)}

    def explain(self, request: Dict) -> str:
        return f"LLM model server named {self.name}"
