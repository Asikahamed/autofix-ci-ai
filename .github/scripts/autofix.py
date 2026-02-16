import os
import sys
import requests
from pathlib import Path

log_file = sys.argv[1]
log = Path(log_file).read_text()

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL = "codellama/CodeLlama-7b-Instruct-hf"

prompt = f"""
You are a CI autofix bot.

Given this pytest failure log, generate the MINIMAL safe code fix.
Return ONLY the corrected Python code.

Pytest log:
{log}
"""

ROUTER_URL = f"https://router.huggingface.co/models/{MODEL}"

def get_ai_fix():
    try:
        response = requests.post(
            ROUTER_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
            timeout=60,
        )
        # If response is empty or API fails, return None
        if not response.text:
            print("Hugging Face returned empty response.")
            return None

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            print("HuggingFace returned unexpected data:", data)
            return None
    except Exception as e:
        print("Error calling Hugging Face API:", e)
        return None

# ✅ Attempt to get AI fix
output = get_ai_fix()

if output and "def " in output:
    Path("calc.py").write_text(output)
    print("calc.py updated by AI")
else:
    print("No valid fix detected or API failed")


