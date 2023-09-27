import commentjson as json

from symbol_library import SymType
from tree import Node


def read_expressions(filepath):
    expressions = []
    with open(filepath, "r") as file:
        for line in file:
            expressions.append(line.strip().split(" "))
    return expressions


def read_expressions_json(filepath):
    with open(filepath, "r") as file:
        return [Node.from_dict(d) for d in json.load(file)]


def tokens_to_tree(tokens, symbols):
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    expr_str = ''.join(tokens)
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in symbols and symbols[token]["type"] in [SymType.Var, SymType.Const, SymType.Literal]:
            out_stack.append(Node(token))
        elif token in symbols and symbols[token]["type"] is SymType.Fun:
            if symbols[token]["precedence"] <= 0:
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif token in symbols and symbols[token]["type"] is SymType.Operator:
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and symbols[operator_stack[-1]]["precedence"] > symbols[token]["precedence"]:
                if symbols[operator_stack[-1]]["type"] is SymType.Fun:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if symbols[operator_stack[-1]]["type"] is SymType.Fun:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols \
                    and symbols[operator_stack[-1]]["type"] is SymType.Fun:
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")

def load_config_file(path):
    with open(path, "r") as file:
        jo = json.load(file)
    return jo
