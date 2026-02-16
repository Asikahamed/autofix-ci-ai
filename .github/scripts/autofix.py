import os
import sys
import requests
from pathlib import Path
import ast
import re

# --- Read CI log artifact
log_file = sys.argv[1]
log = Path(log_file).read_text()

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL = "codellama/CodeLlama-7b-Instruct-hf"

# --- Extract Python files mentioned in log
files_to_fix = set(re.findall(r'File "([^"]+\.py)"', log))
if not files_to_fix:
    # If no files detected, leave as None; AI output will be reviewed but no files overwritten
    files_to_fix = None
    print("No specific Python files detected in CI log. AI output will not overwrite any file automatically.")

# --- Prepare prompt for AI
prompt = f"""
You are a CI autofix bot.

The following files have CI failures: {list(files_to_fix) if files_to_fix else 'Unknown'}.

Given the unified CI log below, generate the MINIMAL safe code fix.
- Only modify the affected files if they are known.
- Do NOT change unrelated code.
- Return ONLY the corrected Python code for the files mentioned.

CI log:
{log}
"""

ROUTER_URL = f"https://router.huggingface.co/models/{MODEL}"

def get_ai_fix():
    """Call Hugging Face API to get AI-generated fix"""
    try:
        response = requests.post(
            ROUTER_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
            timeout=60,
        )
        if not response.text:
            print("Hugging Face returned empty response.")
            return None

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            print("Hugging Face returned unexpected data:", data)
            return None
    except Exception as e:
        print("Error calling Hugging Face API:", e)
        return None

def safe_write(file_path, code):
    """Validate Python code before overwriting"""
    try:
        ast.parse(code)
        Path(file_path).write_text(code)
        print(f"{file_path} updated safely by AI")
    except Exception as e:
        print(f"AI output invalid Python for {file_path}: {e}")

# --- Main logic
output = get_ai_fix()

if output:
    if files_to_fix:
        # Only update files detected in CI log
        for file_path in files_to_fix:
            if file_path in output:
                safe_write(file_path, output)
    else:
        # No files detected: do not overwrite anything
        print("No files detected in log; AI output not written automatically.")
else:
    print("No valid fix detected or API failed")
