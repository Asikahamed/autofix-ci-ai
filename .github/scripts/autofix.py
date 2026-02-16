import os
import sys
import requests
from pathlib import Path
import ast
import re

# --- Read the unified CI log
if len(sys.argv) < 2:
    print("Usage: python autofix.py <ci_log_file>")
    sys.exit(1)

log_file = sys.argv[1]
if not Path(log_file).exists():
    print(f"Log file does not exist: {log_file}")
    sys.exit(1)

log = Path(log_file).read_text()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN not set in environment")
    sys.exit(1)

MODEL = "codellama/CodeLlama-7b-Instruct-hf"
ROUTER_URL = f"https://router.huggingface.co/models/{MODEL}"

# --- Build AI prompt for generic CI failure
prompt = f"""
You are a CI autofix bot.

The CI workflow failed. The unified CI log is provided below:

{log}

Instructions:
- Generate minimal safe fixes to resolve the CI failure.
- This could include: fixing Python code, updating dependencies, correcting configs, or any other step needed to make CI pass.
- Only modify files that the AI identifies as needing a change.
- For Python files, ensure syntax is valid.
- Return the fixed files in a format like:

# File: path/to/file.py
<file contents>

Do NOT include unrelated files.
"""

# --- Call Hugging Face API
def get_ai_fix():
    try:
        print("Calling Hugging Face API...")
        response = requests.post(
            ROUTER_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
            timeout=120,
        )
        print("HF API status code:", response.status_code)
        if not response.text.strip():
            print("Hugging Face returned empty response.")
            return None

        try:
            data = response.json()
        except Exception as e:
            print("Error parsing HF JSON response:", e)
            print("Raw response:", response.text)
            return None

        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            print("Unexpected HF response:", data)
            return None
    except Exception as e:
        print("Error calling Hugging Face API:", e)
        return None

# --- Write AI output safely
def safe_write(file_path, content):
    path = Path(file_path)
    try:
        if file_path.endswith(".py"):
            ast.parse(content)  # Validate Python syntax
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"{file_path} updated safely by AI")
    except Exception as e:
        print(f"Failed to write {file_path}: {e}")

# --- Main logic: parse AI output by file
output = get_ai_fix()

if output:
    # Detect AI-reported files with format "# File: path/to/file"
    file_blocks = re.split(r'# File:\s*([^\n]+)', output)[1:]  # split keeps filenames and content
    # iterate in pairs (filename, content)
    for i in range(0, len(file_blocks), 2):
        file_path = file_blocks[i].strip()
        content = file_blocks[i + 1].lstrip('\n')
        safe_write(file_path, content)
else:
    print("No valid fix detected or API failed")
