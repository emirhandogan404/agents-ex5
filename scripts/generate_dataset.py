import numpy as np
import rootpath
import torch
from random import choice
from datasets import Dataset, load_dataset
from litgpt import LLM
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

rootpath.append(pattern="pyproject.toml")

import project_paths as pp
from src.utils import replace_pattern


def save_dataset(ds):
    """
    Make sure this outputs exactly what you need for the running, filtering, and finetuning to work.
    """
    ds = ds.select_columns(["instruction", "input", "enhanced_output"])
    ds = ds.rename_column("enhanced_output", "output")
    ds.to_json(pp.DATA_DIR / "generated_dataset.json")

# TODO: Complete this function.
def prepare_prompt(sample):
    prompt=sample["output"]
    # randomly select one of the enhancement prompts from pp.DATA_DIR / "prompts"
    # use replace_pattern to fill in the relevant parts (tool_start,...) based on the Config (pp.CONFIG)
    # Hint: the input is the samples "output"
    # Remember the prompt-postfix if you use Qwen!
    # return as "enhancement_prompt"

    prompt = replace_pattern(prompt, "tool_start", pp.CONFIG["tool_start"])
    prompt = replace_pattern(prompt, "tool_end", pp.CONFIG["tool_end"])
    prompt = replace_pattern(prompt, "tool_call", pp.CONFIG["tool_call"])
    prompt = replace_pattern(prompt, "prompt_postfix", pp.CONFIG["prompt_postfix"])

    return {"enhancement_prompt": prompt}


#TODO: complete the function
def enhance_outputs(dataset, llm, tokenizer):
    dataset = list(dataset)
    for i,row in enumerate(tqdm(dataset)):
        prompt = row["enhancement_prompt"]
        # generate a continuation for the prompt using the llm
        # remember to set a reasonable limit for max_new_tokens, and set good parameters for the generation.
        # Hint: use llm.generate
        generated = llm.generate(prompt, max_new_tokens=100, temperature=0.1, top_k=5)
        row["enhanced_output"] = generated
        if i%5==4:
            save_dataset(Dataset.from_list(dataset[:i]))
    return Dataset.from_list(dataset)


def load_prerequisites():
    print("Loading Dataset...")
    ds = load_dataset(
        "json",
        data_files=str(pp.DATA_DIR / pp.CONFIG["dataset"]["name"]),
        split="train",
    )
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(pp.MODEL_CHKPT_DIR / pp.CONFIG["model"])
    llm = LLM.load(
        pp.MODEL_CHKPT_DIR / pp.CONFIG["model"],
        tokenizer_dir=pp.MODEL_CHKPT_DIR / pp.CONFIG["model"],
    )
    print("Done")
    return ds, llm, tokenizer


def main():
    ds, llm, tokenizer = load_prerequisites()
    print("Making Enhancement Prompts...")
    ds = ds.select(range(200)).map(prepare_prompt)
    print("Enhancing Outputs...")
    ds = enhance_outputs(ds, llm, tokenizer)
    for example in ds.select(range(2)):
        print(example["enhanced_output"], end="\n" + "=" * 15 + "\n")
    print("Making Dataset for SFT...")
    save_dataset(ds)
    print("Done.")


if __name__ == "__main__":
    main()
