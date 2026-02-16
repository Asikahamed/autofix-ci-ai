import os
import sys
import requests
import time
from pathlib import Path
import ast
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models that ACTUALLY work on free Hugging Face serverless API
MODELS = {
    # Small, fast models that work reliably
    "phi": "microsoft/phi-2",
    "tiny": "bigcode/tiny_starcoder_py",
    
    # Medium models (may need loading time)
    "starcoder": "bigcode/starcoder2-3b",
    
    # Note: Many popular models (deepseek, llama, etc.) require PRO tier
}

# Use phi-2 (most reliable free model)
MODEL = MODELS["phi"]
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

# Skip AI entirely and use only pattern-based fixes
USE_AI = os.environ.get("USE_AI", "false").lower() == "true"

MAX_RETRIES = 2
RETRY_DELAY = 15

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_log_file(log_path):
    """Read and return CI log contents"""
    if not Path(log_path).exists():
        print(f"❌ Log file does not exist: {log_path}")
        sys.exit(1)
    return Path(log_path).read_text()

def extract_error_context(log):
    """Extract relevant error information from CI log"""
    lines = log.split('\n')
    error_lines = []
    in_error = False
    
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ['error', 'failed', 'traceback', 'assertion', 'test_']):
            in_error = True
            start = max(0, i - 5)
            end = min(len(lines), i + 15)
            error_lines.extend(lines[start:end])
        elif in_error and line.strip() == '':
            in_error = False
    
    return '\n'.join(error_lines) if error_lines else log[:2000]

def build_prompt(log):
    """Build AI prompt for code fixing"""
    error_context = extract_error_context(log)
    
    return f"""Fix this Python test failure.

Error:
{error_context}

Output format:
# File: path/to/file.py
<corrected code>

Example:
# File: app/calc.py
def subtract(a, b):
    return a - b
"""

def call_huggingface_api(prompt):
    """Call Hugging Face Serverless Inference API"""
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.1,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  🔄 Attempt {attempt}/{MAX_RETRIES}...")
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            print(f"  📡 Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "") if isinstance(data[0], dict) else str(data[0])
                if isinstance(data, dict):
                    return data.get("generated_text", "")
                return None
            
            if response.status_code == 503:
                print(f"  ⏳ Model loading...")
                time.sleep(RETRY_DELAY)
                continue
            
            if response.status_code == 429:
                print(f"  ⏳ Rate limited...")
                time.sleep(20)
                continue
            
            print(f"  ❌ Error: {response.text[:200]}")
            
            if response.status_code in [401, 403, 404]:
                return None
            
            if attempt < MAX_RETRIES:
                time.sleep(10)
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
    
    return None

def parse_ai_output(output):
    """Parse AI output and extract file fixes"""
    if not output:
        return []
    
    output = output.strip()
    output = re.sub(r'^```(?:python)?\n', '', output, flags=re.MULTILINE)
    output = re.sub(r'\n```$', '', output)
    
    file_pattern = r'#\s*File:\s*([^\n]+)'
    matches = list(re.finditer(file_pattern, output))
    
    if not matches:
        return []
    
    fixes = []
    for i, match in enumerate(matches):
        file_path = match.group(1).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        content = output[start_pos:end_pos].strip()
        
        content = re.sub(r'^```python\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        
        if content:
            fixes.append({'path': file_path, 'content': content})
    
    return fixes

def validate_python_syntax(content):
    """Validate Python syntax"""
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def apply_fix(file_path, content):
    """Apply fix to file"""
    path = Path(file_path)
    
    if file_path.endswith('.py'):
        valid, error = validate_python_syntax(content)
        if not valid:
            print(f"  ❌ Syntax error: {error}")
            return False
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"  ✅ Fixed {file_path}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def apply_pattern_fixes(log):
    """Pattern-based fallback fixes"""
    print("\n🔧 Applying pattern-based fixes...")
    fixed = False
    
    # Pattern 1: Fix subtract function with wrong operator
    if "test_subtract" in log and ("AssertionError" in log or "assert" in log.lower()):
        for py_file in Path('.').rglob('*.py'):
            if 'test_' in py_file.name or '.backup' in py_file.name:
                continue
            
            try:
                content = py_file.read_text()
                pattern = r'(def\s+subtract\s*\([^)]*\)\s*:\s*return\s+)(\w+)\s*\+\s*(\w+)'
                
                if re.search(pattern, content):
                    fixed_content = re.sub(pattern, r'\1\2 - \3', content)
                    if fixed_content != content:
                        print(f"  🔍 Found bug in {py_file}")
                        apply_fix(str(py_file), fixed_content)
                        fixed = True
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
    
    # Pattern 2: Missing imports
    if "ModuleNotFoundError" in log or "ImportError" in log:
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", log)
        if match:
            print(f"  ⚠️ Missing module: {match.group(1)}")
    
    # Pattern 3: Undefined variable
    if "NameError" in log:
        match = re.search(r"name '([^']+)' is not defined", log)
        if match:
            print(f"  ⚠️ Undefined variable: {match.group(1)}")
    
    return fixed

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🤖 AI Autofix Bot - Generic Python CI Fixer")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("Usage: python autofix.py <ci_log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    print(f"\n📄 Reading: {log_file}")
    log = read_log_file(log_file)
    print(f"📊 Log size: {len(log)} characters")
    
    # Try AI only if explicitly enabled
    if USE_AI and os.environ.get("HF_TOKEN"):
        print(f"\n🤖 Attempting AI fix ({MODEL})...")
        prompt = build_prompt(log)
        
        try:
            output = call_huggingface_api(prompt)
            if output:
                print("\n✅ AI response received")
                fixes = parse_ai_output(output)
                
                if fixes:
                    print(f"\n🔧 Applying {len(fixes)} AI fix(es)...")
                    success = sum(1 for f in fixes if apply_fix(f['path'], f['content']))
                    
                    if success > 0:
                        print(f"\n✅ Applied {success} AI fix(es)")
                        print("=" * 70)
                        sys.exit(0)
        except Exception as e:
            print(f"\n⚠️ AI failed: {e}")
    else:
        print("\n⏭️  Skipping AI (USE_AI=false or no token)")
    
    # Always use pattern-based fixes
    if apply_pattern_fixes(log):
        print("\n✅ Pattern fix applied")
        print("=" * 70)
        sys.exit(0)
    
    print("\n❌ No fixes could be applied")
    print("=" * 70)
    sys.exit(1)

if __name__ == "__main__":
    main()