import ast
import math
import re
import operator
from operator import itemgetter

def calendar(*args):
    from datetime import datetime
    return datetime.now().strftime("Today is %a., %Y-%m-%dT%H:%M:%S")

def qa(query:str):
    """Search the world wide web for pages relevant to the users query.
    Args:
        query (str): The query to search for using DuckDuckGo.
    """
    from ddgs import DDGS
    results = DDGS().text(query, max_results=2)
    op = itemgetter("title", "body")
    results = [op(r) for r in results]
    return results

def wikisearch(query:str):
    """
    Return the summary of the first Wikipedia search result
    in the format "title>summary".
    """
    import wikipedia
    # Get the first search result
    title = wikipedia.search(query, results=1)[0]
    try:
        summary = wikipedia.summary(title, sentences=2)
    except wikipedia.DisambiguationError as e:
        # If disambiguation, take the first option
        summary = wikipedia.summary(e.options[0], sentences=2)
    return f"{title}>{summary}"



# Allowed operators
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg
}

# Allowed functions
functions = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,  # optional extra
}

def calculate(expr: str):
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, %, **, parentheses, sin, cos, tan
    Semi-safe. Use at your own risk.
    """
    def _eval(node):
        if isinstance(node, ast.Constant):  # numbers
            return node.n
        elif isinstance(node, ast.BinOp):  # binary operations
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # unary operations (-x)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](_eval(node.operand))
        elif isinstance(node, ast.Call):  # function calls
            func_name = node.func.id
            if func_name in functions:
                args = [_eval(arg) for arg in node.args]
                return functions[func_name](*args)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr, mode='eval')
    return _eval(tree.body)

import io
import contextlib

def run_with_confirmation(user_code: str, globals_dict=None, locals_dict=None):
    """
    Execute arbitrary Python code using exec(), with confirmation from the observer.
    Captures stdout and stderr, and ensures exceptions do not crash the program.
    Returns a tuple: (globals_dict, stdout_output, stderr_output, error_message)
    """
    print("=== Code Received ===")
    print(user_code)
    print("=====================")

    choice = input("Execute this code? (y/N): ").strip().lower()

    if choice != "y":
        print("Execution cancelled.")
        return None, "", "", None

    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = globals_dict

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    error_message = None

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(user_code, globals_dict, locals_dict)
    except Exception as e:
        error_message = str(e)

    stdout_output = stdout_buffer.getvalue()
    stderr_output = stderr_buffer.getvalue()

    return globals_dict, stdout_output, stderr_output, error_message



def extract_valid_python_code(markdown: str):
    """
    Extract fenced code blocks from markdown, check if they are valid Python,
    and return them as a list.
    """
    code_blocks = []

    # Match fenced code blocks ```python ... ``` or ``` ... ```
    fenced_code_pattern = re.compile(
        r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE
    )

    for match in fenced_code_pattern.findall(markdown):
        code = match.strip()
        if not code:
            continue
        try:
            # Check if code is valid Python
            compile(code, "<string>", "exec")
            code_blocks.append(code)
        except SyntaxError:
            # Skip invalid Python code
            continue

    return code_blocks

def runcode(code:str):
    code_blocks = extract_valid_python_code(code)
    if not code_blocks:
        return "Output: No code has been run."
    _, stdout, _, err_msg = run_with_confirmation(code_blocks[0])
    return f"""Output:\n{stdout}""" + (f"""Error: {err_msg}""" if err_msg else "")


if __name__=="__main__":
    print("Calculator Testing...")
    print("="*15)
    print(calculate("2 + 3 * 4"))       # 14
    print(calculate("(10 - 2) / 4"))    # 2.0
    print(calculate("2 ** 3"))          # 8
    print(calculate("-10 % 3"))          # 2
    print(calculate("sin(0)"))          # 0.0
    print(calculate("cos(0)"))          # 1.0
    try:
        print(calculate("import ddgs"))  # should fail
    except SyntaxError as e:
        print("Expected to fail: ",e)
    
    print("="*15)
    print("QA Testing...")
    print("="*15)
    print(qa("What are primary colors?"))

    print("="*15)
    print("WikiSearch Testing")
    print("="*15)
    print(wiki_search("Python programming"))
    print(wiki_search("Bundeskanzler"))


    print("="*15)
    print("Code Running")
    print("="*15)
    print(run_code("""```
                   import time\nprint(time.time())```"""))