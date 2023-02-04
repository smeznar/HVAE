from enum import Enum


class SymType(Enum):
    Var = 1
    Const = 2
    Operator = 3
    Fun = 4
    Literal = 5


def generate_symbol_library(num_vars, symbol_list, has_constant=True):
    all_symbols = {
        "+": {"symbol": '+', "type": SymType.Operator, "precedence": 0},
        "-": {"symbol": '-', "type": SymType.Operator, "precedence": 0},
        "*": {"symbol": '*', "type": SymType.Operator, "precedence": 1},
        "/": {"symbol": '/', "type": SymType.Operator, "precedence": 1},
        "^": {"symbol": "^", "type": SymType.Operator, "precedence": 2},
        "sqrt": {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5},
        "sin": {"symbol": 'sin', "type": SymType.Fun, "precedence": 5},
        "cos": {"symbol": 'cos', "type": SymType.Fun, "precedence": 5},
        "exp": {"symbol": 'exp', "type": SymType.Fun, "precedence": 5},
        "log": {"symbol": 'log', "type": SymType.Fun, "precedence": 5},
        "^2": {"symbol": '^2', "type": SymType.Fun, "precedence": -1},
        "^3": {"symbol": '^3', "type": SymType.Fun, "precedence": -1},
        "^4": {"symbol": '^4', "type": SymType.Fun, "precedence": -1},
        "^5": {"symbol": '^5', "type": SymType.Fun, "precedence": -1},
    }
    variable_names = 'ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'
    symbols = []
    for i in range(num_vars):
        if i < len(variable_names):
            symbols.append({"symbol": variable_names[i], "type": SymType.Var, "precedence": 5})
        else:
            raise Exception("Insufficient symbol names, please add additional symbols into the variable_names variable"
                            " from the generate_symbol_library method in symbol_library.py")

    if has_constant:
        symbols.append({"symbol": 'C', "type": SymType.Const, "precedence": 5})

    for s in symbol_list:
        if s in all_symbols:
            symbols.append(all_symbols[s])
        else:
            raise Exception(f"Symbol {s} is not in the standard library, please add it into the all_symbols variable"
                            f" from the generate_symbol_library method in symbol_library.py")

    return symbols
