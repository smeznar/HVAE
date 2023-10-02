import sys
sys.path.append('../src')

from symbol_library import generate_symbol_library
from hvae_utils import read_expressions, expression_set_to_json

if __name__ == '__main__':
    # Create a library of symbols that occur in the expressions
    # If your symbol doesn't exist in the "symbol_library.py" and "rusteval/src/evaluator.rs",
    # you can add it yourself or create an issues on the github page
    num_variables = 1
    symbols = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'sqrt', 'log', '^2', '^3', '^4', '^5']
    has_constants = False
    sym_lib = generate_symbol_library(num_variables, symbols, has_constants)

    # Load expressions from the file: here we recommend using the read_expressions method which read a file where each
    # expression is in its line and symbols are split by a whitespace.
    # the expressions variable should be a list of expressions, where an expression is represented as a list of (string) symbols
    expressions = read_expressions("../data/nguyen_expressions.txt")

    output_path = "nguyen1var.json"
    expression_set_to_json(expressions, sym_lib, output_path)

