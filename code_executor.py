# code_executor.py
import subprocess
import sys
import os

def execute_code(code: str) -> (bool, str):
    with open('temp_code.py', 'w') as f:
        f.write(code)
    try:
        result = subprocess.run(
            [sys.executable, 'temp_code.py'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        success = True
    except subprocess.CalledProcessError as e:
        output = e.stderr
        success = False

    os.remove('temp_code.py')

    return success, output
