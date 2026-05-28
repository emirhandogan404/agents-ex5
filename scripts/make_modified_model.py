# %%
import rootpath
import subprocess
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from litgpt import LLM

rootpath.append()

import project_paths as pp

# %%
print("Modify HF model.")
tokenizer = AutoTokenizer.from_pretrained(
    pp.MODEL_CHKPT_DIR / pp.CONFIG["finetune_model"]
)
model = AutoModelForCausalLM.from_pretrained(
    pp.MODEL_CHKPT_DIR / pp.CONFIG["finetune_model"]
)

special_tokens = [
    pp.CONFIG["toolcalling"]["tool_start"],
    pp.CONFIG["toolcalling"]["tool_call"],
    pp.CONFIG["toolcalling"]["tool_end"],
]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

MODIFIED_LOCATION = pp.DATA_DIR / pp.CONFIG["finetune_model"]
model.save_pretrained(MODIFIED_LOCATION)
tokenizer.save_pretrained(MODIFIED_LOCATION)

# %%
# Apply the convert_to_litgpt with the corresponding model name
print("Updating litgpt model...")
cmd = f"""uv run litgpt convert_to_litgpt {MODIFIED_LOCATION} --model_name {pp.CONFIG["finetune_model"]}"""
subprocess.run(cmd, shell=True) # yes this is technically unsafe

# %%
print("Updating Config...")
conf_pth = MODIFIED_LOCATION / "model_config.yaml"
cfg = yaml.safe_load(open(conf_pth))
print("Current Cfg:\n", cfg)
cfg["vocab_size"] = len(tokenizer)
cfg["padded_vocab_size"] = len(tokenizer)
with open(conf_pth, "w") as file:
    yaml.dump(cfg, file)
# %%
print("Verification...")
llm = LLM.load(MODIFIED_LOCATION)
print(
    "This should end with the same token as in the next example:\n",
    llm.tokenizer.encode(
        f"""What are the primary colors?{pp.CONFIG["toolcalling"]["tool_start"]}"""
    )
)
print("This should contain at most two tokens (start-of-sequence & special token):\n",llm.tokenizer.encode(f"""{pp.CONFIG["toolcalling"]["tool_start"]}"""))
print("Verify that your model now uses a _single_ token for your tool_start token in both examples!")
# %%
