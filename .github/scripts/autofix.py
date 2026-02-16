import sys
from openai import OpenAI

log_path = sys.argv[1]

with open(log_path) as f:
    logs = f.read()

client = OpenAI()

prompt = f"""
You are a senior Python engineer.

CI failure logs:
{logs}

Return ONLY the minimal safe code fix.
No explanation.
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}],
)

fix = response.choices[0].message.content.strip()

with open("app/calculator.py", "a") as f:
    f.write("\n# AI FIX\n")
    f.write(fix + "\n")

print("Applied AI fix.")
