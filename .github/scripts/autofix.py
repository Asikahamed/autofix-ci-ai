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

# ✅ Use the new router endpoint
ROUTER_URL = f"https://router.huggingface.co/models/{MODEL}"

response = requests.post(
    ROUTER_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
    timeout=60,
)

data = response.json()

if isinstance(data, list):
    output = data[0]["generated_text"]
else:
    raise RuntimeError(f"HuggingFace error: {data}")

print("=== AI RESPONSE ===")
print(output)

if "def " in output:
    Path("calc.py").write_text(output)
    print("calc.py updated by AI")
else:
    print("No valid fix detected")
