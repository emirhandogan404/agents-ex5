import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    Trainer,
)


def prepare_example(example, tokenizer, tool_call, tool_end):
    example_as_chat = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["instruction"] + example["input"]},
            {"role": "model", "content": example["output"]},
        ], tokenize=False
    )
    encoded = tokenizer(
        example_as_chat,
        truncation=True,
        padding=False,
        max_length=2048,
    )
    encoded["labels"] = encoded["input_ids"].copy()
    encoded["tool_call_id"] = tokenizer.encode(
        tool_call
    )  # pass raw text for loss override
    encoded["tool_end_id"] = tokenizer.encode(
        tool_end
    )  # pass raw text for loss override
    return encoded


class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        This is the normal CrossEntropy Loss, just ignoring everything between
        tool_start and tool_end
        """

        tool_call_id = inputs.get("tool_call_id", None)
        tool_end_id = inputs.get("tool_end_id", None)
        labels = inputs["labels"]

        # Your task: write the code that does the following steps.
        # Build a mask of which samples should be ignored
        # If all are ignored → return zero loss
        # Replace labels with modified labels in the inputs dict

        # Standard LM forward
        outputs = model(**{k: v for k, v in inputs.items() if k != "input_texts"})
        logits = outputs.logits

        # Compute causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss() # Hint: check the documentation for this loss.
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ) # causal-lm loss

        return (loss, outputs) if return_outputs else loss
