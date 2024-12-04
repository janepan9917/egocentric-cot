from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import logging

import logging

@dataclass
class PromptSet():
    """
    Class to hold prompts to query model

    """
    prompt_type: str
    sys_prompt: str
    query_prompt: str
    prefill: str = None

def get_prompts(
    config: Dict[str, str],
    prompt_type: str, 
    prompt_args: Dict[str, str] = {},
) -> PromptSet:
    """
    Get prompt from config file and fill in with relevant info, if needed.

    Attributes
    ----------
    prompt_type: str
        Type of prompt. Can be one of the following:
        - direct
        - cot
    prompt_config : Dict[str, str]
        Dictionary with base prompt templates
    prompt_args : Dict[str, str]
        Dictionary with relevant info to fill into prompt templates.
    
    """

    sys = config["sys"].format(**prompt_args)
    query = config[f"query_{prompt_type}"].format(**prompt_args)
    prefill = config[f"prefill_{prompt_type}"].format(**prompt_args)
    
    logging.warning("Stripping newlines from prompts.")
    return PromptSet(
        prompt_type=prompt_type,
        sys_prompt=sys.strip(),
        query_prompt=query.strip(),
        prefill=prefill.strip(),
    )
        
