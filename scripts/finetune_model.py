import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import rootpath

rootpath.append(pattern="pyproject.toml")
import project_paths as pp
from src.train import CustomLossTrainer, prepare_example

MODEL_NAME = pp.DATA_DIR / pp.CONFIG["finetune_model"]
DATASET_FILE = pp.DATA_DIR / "finetune_dataset.json"


def load_json_dataset(path):
    # Must return a HF dataset with "instruction" and "output" or "text"
    return load_dataset("json", data_files=str(path), split="train")


def main():
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load and preprocess data
    # raw_ds = load_json_dataset(DATASET_FILE)
    raw_ds = load_json_dataset(DATASET_FILE).select(range(10)) # for testing we only load a small subset
    tokenized_ds = raw_ds.map(lambda e: prepare_example(e,tokenizer, pp.CONFIG["toolcalling"]["tool_call"], pp.CONFIG["toolcalling"]["tool_end"]), batched=False)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=pp.PLOTS_DIR / pp.CONFIG["finetune_model"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=20,
        save_steps=500,
        # fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(pp.PLOTS_DIR / pp.CONFIG["finetune_model"])

if __name__=="__main__":
    main()