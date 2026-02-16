import os
import sys
import requests
import time
from pathlib import Path
import ast
import re
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Free models that work well (choose one):
MODELS = {
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",  # Good for code
    "starcoder": "bigcode/starcoder",                    # Good for code
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",    # Good for instructions
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",  # Best for code fixes
}

# Use DeepSeek Coder (best free option for code fixes)
MODEL = MODELS["deepseek"]
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
API_URL = f"https://router.huggingface.co/models/{MODEL}/v1/chat/completions"

MAX_RETRIES = 3
RETRY_DELAY = 20  # seconds (for model loading)

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
    
    # Find test failures
    error_lines = []
    in_error = False
    
    for i, line in enumerate(lines):
        # Detect error sections
        if any(keyword in line.lower() for keyword in ['error', 'failed', 'traceback', 'assertion']):
            in_error = True
            # Get context (5 lines before and after)
            start = max(0, i - 5)
            end = min(len(lines), i + 10)
            error_lines.extend(lines[start:end])
        elif in_error and line.strip() == '':
            in_error = False
    
    return '\n'.join(error_lines) if error_lines else log

def build_prompt(log):
    """Build AI prompt for code fixing"""
    error_context = extract_error_context(log)
    
    return f"""You are an expert Python developer fixing CI test failures.

CI Error Log:
```
{error_context}
```

Task:
1. Analyze the error and identify the root cause
2. Generate ONLY the minimal fix needed
3. Output corrected files in this exact format:

# File: path/to/file.py
<complete corrected file content>

# File: path/to/another.py
<complete corrected file content>

Rules:
- Only output files that need fixing
- Include complete file content (not just the changed part)
- Ensure valid Python syntax
- Be minimal - don't change unrelated code
- Common fixes: wrong operators (+ vs -), logic errors, missing imports, typos

Example output:
# File: app/calc.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

def call_huggingface_api(prompt):
    """Call Hugging Face Inference API with retries"""
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.1,  # Low temperature for deterministic fixes
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🔄 API Call attempt {attempt}/{MAX_RETRIES}...")
            
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            print(f"📡 Status: {response.status_code}")
            
            # Handle model loading
            if response.status_code == 503:
                error_data = response.json()
                if "loading" in str(error_data).lower():
                    estimated_time = error_data.get("estimated_time", RETRY_DELAY)
                    print(f"⏳ Model loading... waiting {estimated_time}s")
                    time.sleep(estimated_time)
                    continue
            
            # Handle rate limiting
            if response.status_code == 429:
                print("⏳ Rate limited, waiting 30s...")
                time.sleep(30)
                continue
            
            # Success
            if response.status_code == 200:
                data = response.json()
                return extract_generated_text(data)
            
            # Other errors
            print(f"❌ API Error {response.status_code}: {response.text}")
            
            if attempt < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            
        except requests.exceptions.Timeout:
            print(f"⏱️ Request timeout (attempt {attempt})")
            if attempt < MAX_RETRIES:
                time.sleep(10)
        except Exception as e:
            print(f"❌ Exception: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
    
    return None

def extract_generated_text(data):
    """Extract generated text from various response formats"""
    try:
        # Format 1: [{"generated_text": "..."}]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data[0], str):
                return data[0]
        
        # Format 2: {"generated_text": "..."}
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        
        # Format 3: [{"generated_text": "...", "text": "..."}]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return data[0].get("text") or data[0].get("generated_text")
        
        print(f"⚠️ Unexpected response format: {type(data)}")
        return None
        
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return None

def parse_ai_output(output):
    """Parse AI output and extract file fixes"""
    if not output:
        return []
    
    # Clean up the output
    output = output.strip()
    
    # Remove markdown code fences
    output = re.sub(r'^```(?:python)?\n', '', output, flags=re.MULTILINE)
    output = re.sub(r'\n```$', '', output)
    
    # Find all file blocks
    file_pattern = r'#\s*File:\s*([^\n]+)'
    matches = list(re.finditer(file_pattern, output))
    
    if not matches:
        print("⚠️ No file markers found in AI output")
        print("Output preview:", output[:300])
        return []
    
    fixes = []
    for i, match in enumerate(matches):
        file_path = match.group(1).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        content = output[start_pos:end_pos].strip()
        
        # Remove any remaining markdown
        content = re.sub(r'^```python\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        
        if content:
            fixes.append({
                'path': file_path,
                'content': content
            })
    
    return fixes

def validate_python_syntax(content):
    """Validate Python syntax"""
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def apply_fix(file_path, content):
    """Safely write fixed content to file"""
    path = Path(file_path)
    
    # Validate syntax for Python files
    if file_path.endswith('.py'):
        valid, error = validate_python_syntax(content)
        if not valid:
            print(f"❌ Syntax error in {file_path}: {error}")
            return False
    
    try:
        # Create directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup original file
        if path.exists():
            backup_path = path.with_suffix(path.suffix + '.backup')
            backup_path.write_text(path.read_text())
            print(f"💾 Backed up to {backup_path}")
        
        # Write fixed content
        path.write_text(content)
        print(f"✅ Fixed {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to write {file_path}: {e}")
        return False

def apply_pattern_based_fixes(log):
    """Fallback: Apply common pattern-based fixes"""
    print("\n🔧 Applying pattern-based fixes...")
    
    fixed = False
    
    # Pattern 1: subtract function using + instead of -
    if "test_subtract" in log and "AssertionError" in log:
        for py_file in Path('.').rglob('*.py'):
            content = py_file.read_text()
            
            # Fix subtract with wrong operator
            pattern = r'def subtract\([^)]+\):\s*return\s+(\w+)\s*\+\s*(\w+)'
            if re.search(pattern, content):
                fixed_content = re.sub(
                    pattern,
                    r'def subtract(\1, \2):\n    return \1 - \2',
                    content
                )
                if fixed_content != content:
                    apply_fix(str(py_file), fixed_content)
                    fixed = True
    
    # Pattern 2: Missing imports
    if "ModuleNotFoundError" in log or "ImportError" in log:
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", log)
        if match:
            module = match.group(1)
            print(f"⚠️ Missing module: {module}")
            print(f"💡 Add to requirements.txt: {module}")
    
    # Pattern 3: Undefined variable
    if "NameError" in log:
        match = re.search(r"name '([^']+)' is not defined", log)
        if match:
            var = match.group(1)
            print(f"⚠️ Undefined variable: {var}")
    
    return fixed

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("🤖 AI Autofix Bot - Generic Python CI Fixer")
    print("=" * 70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python autofix.py <ci_log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Check HF token
    if not os.environ.get("HF_TOKEN"):
        print("❌ HF_TOKEN not set in environment")
        print("💡 Set it with: export HF_TOKEN='your_token'")
        sys.exit(1)
    
    # Read log
    print(f"\n📄 Reading log: {log_file}")
    log = read_log_file(log_file)
    print(f"📊 Log size: {len(log)} characters")
    
    # Build prompt
    print("\n🔨 Building AI prompt...")
    prompt = build_prompt(log)
    
    # Call AI
    print(f"\n🤖 Calling Hugging Face API ({MODEL})...")
    output = call_huggingface_api(prompt)
    
    if output:
        print("\n📝 AI Response received")
        print("Preview:", output[:200], "...\n")
        
        # Parse fixes
        fixes = parse_ai_output(output)
        
        if fixes:
            print(f"\n✨ Found {len(fixes)} file(s) to fix:")
            for fix in fixes:
                print(f"  - {fix['path']}")
            
            # Apply fixes
            print("\n🔧 Applying fixes...")
            success_count = 0
            for fix in fixes:
                if apply_fix(fix['path'], fix['content']):
                    success_count += 1
            
            print(f"\n✅ Applied {success_count}/{len(fixes)} fixes successfully")
            
            if success_count > 0:
                print("\n" + "=" * 70)
                print("✅ Autofix Complete - Changes Applied")
                print("=" * 70)
                sys.exit(0)
        else:
            print("\n⚠️ No valid fixes found in AI output")
    else:
        print("\n❌ AI API call failed")
    
    # Fallback to pattern-based fixes
    print("\n🔄 Attempting pattern-based fixes...")
    if apply_pattern_based_fixes(log):
        print("\n✅ Pattern-based fix applied")
        sys.exit(0)
    else:
        print("\n❌ No fixes could be applied")
        sys.exit(1)

if __name__ == "__main__":
    main()