from argparse import ArgumentParser
import json

import numpy as np
from ProGED.generators import GeneratorGrammar

from tree import Node
from hvae_utils import load_config_file
from symbol_library import generate_symbol_library, SymType


def generate_grammar(symbols):
    grammar = ""
    operators = {}
    functions = []
    powers = []
    variables = []
    constants = False
    for symbol in symbols:
        if symbol["type"].value == SymType.Operator.value:
            if symbol["precedence"] in operators:
                operators[symbol["precedence"]].append(symbol["symbol"])
            else:
                operators[symbol["precedence"]] = [symbol["symbol"]]
        elif symbol["type"].value == SymType.Fun.value and symbol["precedence"] < 0:
            powers.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Fun.value:
            functions.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Var.value:
            variables.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Const.value:
            constants = True
        else:
            raise Exception("Error during generation of the grammar")

    if 0 in operators:
        op_prob = 0.4 / len(operators[0])
        for op in operators[0]:
            grammar += f"E -> E '{op}' F [{op_prob}]\n"
        grammar += "E -> F [0.6]\n"
    else:
        grammar += "E -> F [1.0]\n"

    if 1 in operators:
        op_prob = 0.4 / len(operators[1])
        for op in operators[1]:
            grammar += f"F -> F '{op}' T [{op_prob}]\n"
        grammar += "F -> T [0.6]\n"
    else:
        grammar += "F -> T [1.0]\n"

    remaining = 1
    if len(powers) > 0:
        grammar += "T -> '(' E ')' P [0.2]\n"
        remaining -= 0.2

    if len(functions) > 0:
        grammar += "T -> R '(' E ')' [0.2]\n"
        remaining -= 0.2

    remaining /= 3
    grammar += f"T -> V [{2*remaining}]\n"
    grammar += f"T -> '(' E ')' [{remaining}]\n"

    var_prob = 1/len(variables) if not constants else 1/(len(variables)+2)
    for v in variables:
        grammar += f"V -> '{v}' [{var_prob}]\n"

    if constants:
        grammar += f"V -> 'C' [{2*var_prob}]\n"

    function_prob = 1/len(functions)
    for funct in functions:
        grammar += f"R -> '{funct}' [{function_prob}]\n"

    powers = sorted(powers)
    power_probs = [1/(1+int(p[1:])) for p in powers]
    power_probs = np.array(power_probs) / sum(power_probs)
    for p, prob in zip(powers, power_probs):
        grammar += f"P -> '{p}' [{prob}]\n"

    return grammar


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens, symbols):
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
    start_expr = "".join(tokens)
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in symbols and (symbols[token]["type"].value == SymType.Var.value or symbols[token]["type"].value == SymType.Const.value) or is_float(token):
            out_stack.append(Node(token))
        elif token in symbols and symbols[token]["type"].value == SymType.Fun.value:
            if token[0] == "^":
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif token in symbols and symbols[token]["type"].value == SymType.Operator.value:
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and symbols[operator_stack[-1]]["precedence"] >= symbols[token]["precedence"]:
                if symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols and symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) < num_tokens:
        raise Exception(f"Could not parse the whole expression {start_expr}")
    return out_stack[-1]


def generate_expressions(grammar, number_of_expressions, symbol_objects, has_constants=True, max_depth=7):
    generator = GeneratorGrammar(grammar)
    expression_set = set()
    expressions = []
    while len(expression_set) < number_of_expressions:
        if len(expression_set) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_set)}")
        expr = generator.generate_one()[0]
        if has_constants:
            pass

        expr_str = "".join(expr)
        if expr_str in expression_set:
            continue

        try:
            expr_tree = tokens_to_tree(expr, symbol_objects)
            if expr_tree.height() > max_depth:
                continue
            # print(expr_str)
            expressions.append(expr_tree)
            expression_set.add(expr_str)
        except:
            continue

    return expressions


if __name__ == '__main__':
    parser = ArgumentParser(prog='Expression set generation', description='Generate a set of expressions')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    Node.add_symbols(sy_lib)
    so = {s["symbol"]: s for s in sy_lib}

    # Optional (recommended): Generate training set from a custom grammar
    grammar = None

    if grammar is None:
        grammar = generate_grammar(sy_lib)

    # print(grammar)

    expressions = generate_expressions(grammar, es_config["num_expressions"], so, expr_config["has_constants"], max_depth=es_config["max_tree_height"])

    expr_dict = [tree.to_dict() for tree in expressions]

    save_path = es_config["expression_set_path"]
    if save_path != "":
        with open(save_path, "w") as file:
            json.dump(expr_dict, file)
