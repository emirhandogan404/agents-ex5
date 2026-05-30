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

        if tool_call_id is None or tool_end_id is None:
            tokenizer = self.data_collator.tokenizer
            import rootpath
            rootpath.append(pattern="pyproject.toml")
            import project_paths as pp
            tc_id = tokenizer.encode(pp.CONFIG["toolcalling"]["tool_call"])[-1]
            te_id = tokenizer.encode(pp.CONFIG["toolcalling"]["tool_end"])[-1]
            tool_call_id = torch.tensor([[tc_id]] * labels.size(0), device=labels.device)
            tool_end_id = torch.tensor([[te_id]] * labels.size(0), device=labels.device)

        # Your task: write the code that does the following steps.
        # Build a mask of which samples should be ignored
        # If all are ignored → return zero loss
        # Replace labels with modified labels in the inputs dict
        all_ignored = False
        if tool_call_id is not None and tool_end_id is not None:
            labels_cpu = labels.cpu()

            for i in range(labels_cpu.size(0)):
                turn_off = False
                for j in range(labels_cpu.size(1)):
                    if labels_cpu[i][j] == tool_call_id[i][-1].item():
                        turn_off = True
                        continue
                    if labels_cpu[i][j] == tool_end_id[i][-1].item():
                        turn_off = False
                    if turn_off:
                        labels_cpu[i][j] = -100
            all_ignored = (labels_cpu == -100).all()

            labels = labels_cpu.to(labels.device)

            inputs["labels"] = labels

        # Standard LM forward
        outputs = model(**{k: v for k, v in inputs.items() if k != "input_texts"})
        logits = outputs.logits

        # Compute causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if not all_ignored:
            loss_fct = CrossEntropyLoss(ignore_index=-100) # Hint: check the documentation for this loss.
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ) # causal-lm loss
        else:
            loss = logits.sum()*0.0

        return (loss, outputs) if return_outputs else loss
