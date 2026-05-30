import re
from datasets import load_dataset
import rootpath

rootpath.append(pattern="pyproject.toml")

import project_paths as pp
from src.tools import qa, wikisearch, calculate, calendar, runcode


import re
from typing import Callable, List, Optional

# the following regex matches:
# ^\s* := beginning of string and any number of "whitespace" characters (including newline)
# ([A-Za-z_]\w*) := capture group 1; containing any number of characters or underscores; does _not_ support numbers
# \s* := any number of whitespace characters
# \(\s*(.*)\s*\) := the second capture group; containing anything delimited by the outermost two parantheses
# \s*$ := any number of whitespace characters until the end of the string
FUNC_CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*(.*)\s*\)\s*$", re.DOTALL)


#TODO: complete this function.
def process_tool_calls(
    text: str,
    start_tag: str,
    end_tag: str,
    call_tag: str,
    handlers: List[Callable[[str], str]],
    default_handler: Optional[Callable[[str, str], str]] = None,
) -> str:
    """
    Process all tool calls delimited by start_tag and end_tag in `text`.
    For each occurrence:
      - extract the content between start_tag and end_tag
      - parse `func_name(argument)` from the content,
      - call handlers[func_name](argument) (or default_handler(func_name, argument)),
      - insert insert_tag + result immediately before the end_tag.

    Returns the transformed text.
    """
    # Escape tags for use in regex matching
    start_esc = re.escape(start_tag)
    end_esc = re.escape(end_tag)

    handlers_dict = {f.__name__: f for f in handlers} # extract names of all the functions

    # Find each start/end pair (non-greedy)
    # This matches _everything_ delimited by start_tag and end_tag
    pair_re = re.compile(rf"{start_esc}(.*?){end_esc}", re.DOTALL)

    out_parts = []
    last_pos = 0

    for m in pair_re.finditer(text):
        # append text before current match
        # you can use m.group(n) with n being an int, to get the "captured" strings in the current match

        # parse function call from body
        # if a function was extracted
        # get appropriate handler
        ## if the handler exists: run it with the exzracted argument, ensure we have a string to insert
        ## if it does not exist and we have a default handler, call that one instead
        ## otherwise make the result: "<parse_error>"
        out_parts.append(text[last_pos:m.start()])

        try: 
            match = FUNC_CALL_RE.search(m.group(1))
            if not match:
                res = "<parse_error>"
            else:
                func_name = match.group(1)
                arg_str = match.group(2).strip()
                
                if len(arg_str) >= 2 and ((arg_str[0] == '"' and arg_str[-1] == '"') or (arg_str[0] == "'" and arg_str[-1] == "'")):
                    arg = arg_str[1:-1]
                else:
                    arg = arg_str

                if func_name in handlers_dict:
                    res = str(handlers_dict[func_name](arg))
                elif default_handler:
                    res = str(default_handler(func_name, arg))
                else:
                    res = "<parse_error>"

            new = m.group(0).removesuffix(end_tag)
            new = new + call_tag + res + end_tag
            out_parts.append(new)
        except Exception:
            new = m.group(0).removesuffix(end_tag)
            new = new + call_tag + "<parse_error>" + end_tag
            out_parts.append(new)
        
        last_pos = m.end()
        # reconstruct the matched segment inserting the result right before the end_tag
        # Hint: the full match (m.group(0)) ends with the literal end_tag, so slice it off
        # append the constructed output to out_parts

    out_parts.append(text[last_pos:])
    # dont forget to append remaining text after last match 
    # Hint: you need to keep track of the position in the loop above 
    # Hint2: m.end() gives you the position of the end of your match
    return "".join(out_parts)

def apply_tool_call_processing(row):
    handlers = [qa, calculate, calendar, wikisearch, runcode]
    new_output = process_tool_calls(
        row["output"],
        pp.CONFIG["toolcalling"]["tool_start"],
        pp.CONFIG["toolcalling"]["tool_end"],
        pp.CONFIG["toolcalling"]["tool_call"],
        handlers,
        default_handler=lambda fn, arg: f"<unknown-fn:{fn}>",
    )
    return {"output":new_output}

def main():
    # load the dataset
    dataset = load_dataset(
        "json", data_files=str(pp.DATA_DIR / "generated_dataset.json"), split="train"
    )
    # apply processing
    dataset = dataset.map(apply_tool_call_processing)
    # re-emit dataset
    for e in dataset.select(range(2)):
        print("="*20)
        print(e)
    dataset.to_json(pp.DATA_DIR / "generated_dataset_after_run.json")


if __name__ == "__main__":
    main()
