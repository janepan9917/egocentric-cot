from dataclasses import dataclass, field
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Optional, Any
import os
# from vllm import LLM, SamplingParams
import argparse
from tenacity import retry, wait_exponential, stop_after_attempt
import torch

from prompt import PromptSet


@dataclass
class Model():
    """
    Handles the actual generation of text from a model.

    Attributes
    ----------
    model : typing.Any
        The actual model itself. You should be able to call self.model.generate() or
        self.model.messages.create() to directly get output from the model.
    tokenizer : typing.Any
        The tokenizer for the model.
        

    Methods
    -------
    load(): 
        Prepares model and tokenizer for generation.
        This is called in get_model() after instantiating the class.
    
    generate(prompts: PromptSet, n_samples: int) -> List[str]:
        Generates text given the query and system prompts.

    """
    config: Dict[str, str] 
    model: 'typing.Any' = None
    tokenizer: 'typing.Any' = None

    def load(self, key):
        raise NotImplementedError("Subclass must implement abstract method")

    def generate(self, prompts: PromptSet, n_samples: int = 1) -> list[str]:
        raise NotImplementedError("Subclass must implement abstract method")


@dataclass
class OpenAIModel(Model):

    def load(self, key: str) -> None:
        self.model = OpenAI(
            api_key=key,
        )
        self.name = self.config["model_name"]

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: PromptSet, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        query_prompt = query_prompt + "\n\n" + prompts.prefill # Prefill isn't implemented for OAI yet

        if sys_prompt == None:
            response = self.model.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "user", "content": query_prompt},
                ],
                n=n_samples,
            )
        else:
            response = self.model.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": query_prompt},
                ],
                n=n_samples,
            )

        return [response.choices[i].message.content for i in range(len(response.choices))]

@dataclass
class ClaudeModel(Model):
    full_model_names = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    }

    def load(self, key: str) -> None:
        self.name = self.full_model_names[self.config.model_name]
        self.model = Anthropic(
            api_key=key,
        )

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(30))
    def generate(
        self, 
        prompts: PromptSet, 
        n_samples: int = 1, 
    ) -> List[str]:
        """ Gets a responses from the Anthropic model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        prefill = prompts.prefill
        
        # populate messages
        print(f"'{prefill}'")
        messages = [{"role": "user", "content": query_prompt}]
        if prefill != "":
            messages.append({"role": "assistant", "content": prefill}) 

        responses = []
        max_tokens = 2056

        for i in range(n_samples):
            if sys_prompt is None:
                response = self.model.messages.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=max_tokens, 

                )
            else:
                response = self.model.messages.create(
                    model=self.name,
                    system= sys_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                
            response = response.content[0].text
            responses.append(response)

        return responses
    
    

@dataclass
class HuggingFaceModel(Model):
    config: Dict = None
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    _device: Optional[torch.device] = None

    def __post_init__(self):
        self._device = torch.device(self.config.device)
        logging.info(f"Using device: {self._device}")

    def load(self, key: Optional[str] = None) -> None:
        """
        Loads the Hugging Face model and tokenizer based on the model name provided in config.
        """
        # Load the Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        logging.info(f"Loaded tokenizer for model {self.config.model_name}")

        # Load the Hugging Face model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Move model to device and set evaluation mode
        self.model.to(self._device)
        self.model.eval()
        
        # Set default generation config
        generation_config = GenerationConfig(**self.config.generation_config.to_dict())
        self.model.generation_config = generation_config
        
        logging.info(f"Loaded model {self.config.model_name} to {self._device}")


    def generate(
        self,
        prompts: PromptSet,
        n_samples: int = 1,
        sampling_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate responses using the model.
        
        Args:
            prompts: PromptSet containing system prompt, query prompt, and optional prefill
            n_samples: Number of samples to generate
            sampling_config: Optional override for generation configuration
        """
        messages = [
            {"role": "system", "content": prompts.sys_prompt},
            {"role": "user", "content": prompts.query_prompt},
        ]

        if prompts.prefill:
            messages.append({"role": "assistant", "content": prompts.prefill})

        # Prepare inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=bool(prompts.prefill),
            add_generation_prompt=not bool(prompts.prefill),
        ).to(self._device)

        # Set up generation config
        if sampling_config:
            generation_config = GenerationConfig(**sampling_config)
        else:
            generation_config = self.model.generation_config

        # Set up stopping criteria
        stop_criteria = None
        if not self.config.stopping_condition is None:
            max_tokens = self.config.stopping_condition['n_tokens']
            max_newlines = self.config.stopping_condition['n_lines']
            
            if max_newlines is not None:
                stop_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
                print(f'Using for newline token id {stop_token_id}')
                stop_criteria = [MaxNewLinesCriteria(
                    inputs.input_ids.shape[1],
                    max_newlines,
                    stop_token_id
                )]
            
            if max_tokens is not None:
                generation_config.max_new_tokens = max_tokens

        import pdb; pdb.set_trace()
        # Generate responses
        response_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model.generate(
                    inputs.input_ids,
                    generation_config=generation_config,
                    stopping_criteria=stop_criteria,
                )

                # Move outputs to CPU if they are on GPU
                if self._device.type == 'cuda':
                    outputs = outputs.cpu()
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_list.append(response)

        return response_list


    



#--------------------------------------------#


OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]

ANTHROPIC_MODEL_NAMES = [
    "claude-3.5-sonnet",
]

# LLAMA_MODEL_NAMES = [
#     "llama-3-8b",
#     "llama-3-8b-instruct",
#     "codellama-7b-instruct",
# ]

# VLLM_LLAMA_MODEL_NAMES = [
#     "vllm-llama-3-8b",
#     "vllm-llama-3-8b-instruct",
# ]
 
#GEMMA_MODEL_NAMES = [
#     "google/gemma-2-27b-it",
#     "princeton-nlp/gemma-2-9b-it-SimPO",
# ]

def get_model(config: Dict[str, str]) -> Model:
    model_name = config["model_name"]
    api_key = config["api_key"]

    if model_name in OPENAI_MODEL_NAMES:
        model = OpenAIModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in ANTHROPIC_MODEL_NAMES:
        model = ClaudeModel(config)
        model.load(os.getenv(api_key))
        return model

    elif model_name in VLLM_LLAMA_MODEL_NAMES:
        model = VLLMModel(config)
        model.load()
        return model

    else:
        raise ValueError(f"Model {model_name} not found")

