#!/usr/bin/env bash
# SETUP ENVIRONMENT
gitroot=$(git rev-parse --show-toplevel)
CHECKPOINT_DIR=$gitroot/../checkpoints

# @student: we suggest using one of these models, we tested them to work on a single A100 with 80GB Mem.
# Feel free to use a different one, by replacing the path in the serve command with the huggingface name of the model.
MODEL=microsoft/phi-4
# MODEL=google/gemma-3-12b-it
uv run litgpt chat $CHECKPOINT_DIR/$MODEL --multiline true
