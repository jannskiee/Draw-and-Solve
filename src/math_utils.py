import re
from sympy import sympify


def safe_evaluate(expr):
    """
    Safely evaluates a mathematical expression using SymPy.
    """
    # Normalize multiplication symbols
    expr = expr.replace("x", "*").replace("X", "*")

    # Validate expression to contain only numbers and allowed operators
    if not re.match(r'^[\d\s+\-*/^().]+$', expr):
        print("Invalid expression\n")
        return None
    try:
        safe_result = float(sympify(expr))
        # Format result: integer if whole number, otherwise truncate decimals to 2 places
        if safe_result.is_integer():
            safe_result = int(safe_result)
        else:
            safe_result = "{:.2f}".format(safe_result).rstrip('0').rstrip('.')

        print(f"Result: {safe_result}\n")
        return str(safe_result)
    except:
        print("Error in evaluation\n")
        return None