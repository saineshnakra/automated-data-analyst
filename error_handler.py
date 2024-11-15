from code_generator import generate_ml_code
from models import GeneratedCode

def handle_error(df, error_message, current_code):
    new_code = generate_ml_code(df, error_message=error_message, current_code=current_code)
    return new_code
