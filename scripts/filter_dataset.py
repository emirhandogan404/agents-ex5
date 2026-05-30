import math
import torch
from litgpt import LLM  # (or whatever import you use)
from tqdm import tqdm
from datasets import load_dataset, Dataset
import rootpath

rootpath.append()

import project_paths as pp


def load_model():
    llm = LLM.load(pp.MODEL_CHKPT_DIR / pp.CONFIG["model"])
    tokenizer = llm.tokenizer  # or however you access the tokenizer
    return tokenizer, llm

# TODO: complete this function.
@torch.no_grad()
def calculate_loss(llm, tokenizer, sample):
    """
    Calculate the perplexity loss for a sample using the LitGPT library and torch.
    """
    # get the sample converted to tokens then get the logits for that sample
    # Hint: fix the shape after applying the encode function of the sample

    input_ids = tokenizer.encode(sample).to(llm.model.device)
    input_ids = input_ids.unsqueeze(0)
    logits = llm.model(input_ids)

    # Shift the logits and input_ids so that at position t we predict token t
    # make sure to make the tensor "contigous in memory"
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous().to(torch.long)

    # Use cross-entropy (or negative log-likelihood)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # correctly apply crossentropy to the flattened logits & labels
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # then reshape back to (batch, seq_len)
    loss = loss.view(-1)
    # Calculate average negative log-likelihood (avg_nll) per token
    avg_nll = loss.mean()
    # Perplexity as the exponential of the avg_nll
    ppl = torch.exp(avg_nll)
    return ppl.item()

def find_tool_call_from(text, offset, tool_start, tool_call, tool_end):
    sub_text = text[offset:]
    tool_start_idx = sub_text.find(tool_start)
    tool_call_idx = sub_text.find(tool_call)
    tool_end_idx = sub_text.find(tool_end)
    if (tool_start_idx == -1) or (tool_call_idx == -1) or (tool_end_idx == -1):
        return None
    return (tool_start_idx+offset,tool_call_idx+offset,tool_end_idx+offset)


# TODO: complete this function
def filter_api_calls(examples, llm, tokenizer, tool_start, tool_call, tool_end, threshold=1.0):
    """
    Args:
        - examples: hf.Dataset of the examples from the previous step
        - llm: the litgpt LLM model
        - tokenizer: the corresponding tokenizer
        - tool_start,tool_call,tool_end: the actual strings for the new special tokens
        - threshold: when is the improvement big enough, should be between 0.5 and 2.0 acc. to paper
    """
    examples = list(examples)
    for row in tqdm(examples):
        text = row["output"]
        
        # calculate the overall positive loss Lp (see Toolformer paper)
        Lp = calculate_loss(llm, tokenizer, text)
        
        offset = 0
        result = ""
        
        # while you can extract another tool call (find_tool_call_from)
        while True:
            match = find_tool_call_from(text, offset, tool_start, tool_call, tool_end)
            if not match:
                result += text[offset:]
                break
                
            start_idx, call_idx, end_idx = match
            
            ## calculate Ln as the min of:
            ## 1. no API call at all
            text_no_api = text[:start_idx] + text[end_idx + len(tool_end):]
            loss_no_api = calculate_loss(llm, tokenizer, text_no_api)
            
            ## 2. API call but no result provided
            text_no_result = text[:call_idx + len(tool_call)] + text[end_idx + len(tool_end):]
            loss_no_result = calculate_loss(llm, tokenizer, text_no_result)
            
            Ln = min(loss_no_api, loss_no_result)
            
            ## if the Lp is >threshold much better than Ln, keep the api call
            if (Ln - Lp) >= threshold:
                result += text[offset:end_idx + len(tool_end)]
            else:
                result += text[offset:start_idx]
                
            offset = end_idx + len(tool_end)
            
        # add the resulting text as the "output_filtered" column
        row["output_filtered"] = result
    return Dataset.from_list(examples)


def main():
    tokenizer, llm = load_model()
    ds = load_dataset("json", data_files=str(pp.DATA_DIR / "generated_dataset_after_run.json"), split="train")
    ds = filter_api_calls(
        ds,
        llm,
        tokenizer,
        pp.CONFIG["toolcalling"]["tool_start"],
        pp.CONFIG["toolcalling"]["tool_call"],
        pp.CONFIG["toolcalling"]["tool_end"],
    )
    ds = ds.select_columns(["instruction", "input", "output_filtered"])
    ds = ds.rename_column("output_filtered", "output")
    ds.to_json(pp.DATA_DIR / "finetune_dataset.json")


if __name__ == "__main__":
    main()
