from src.fasteval.functions import create_tokens
from src.fasteval.program import Program
from src.fasteval.library import Library
from src.symbol_library import SymType
# from src.symbol_library import generate_symbol_library
# from src.tree import Node
# from src.utils import tokens_to_tree

import numpy as np


def symbols_to_psymbols(symbols):
    psymbols = []
    for s in symbols:
        if s["type"].value in [SymType.Operator.value, SymType.Fun.value]:
            psymbols.append(s["psymbol"])
    return psymbols


class FastEval:
    def __init__(self, X, num_vars, function_set, has_const=True, protected=False, invalid=1e10):
        function_set = symbols_to_psymbols(function_set)
        if has_const:
            function_set.append("const")
        tokens = create_tokens(num_vars, function_set, protected)
        library = Library(tokens)
        Program.set_execute(protected)
        Program.library = library
        Program.set_const_optimizer("scipy", **{})
        FastEval.X = X
        FastEval.invalid_reward = invalid

    def execute(self, tree):
        pexpr = tree.to_pexpr()
        p = Program(pexpr)
        p.optimize(FastEval.rmse)
        return p.execute(FastEval.X), p.get_constants()

    @staticmethod
    def rmse(p):
        y_hat = p.execute(FastEval.X)

        if p.invalid:
            return FastEval.invalid_reward

        # Compute metric
        r = np.sqrt(np.square(np.subtract(FastEval.X[:, -1], y_hat)).mean())

        return r


if __name__ == '__main__':
    symbols = generate_symbol_library(1, ["+", "-", "*", "/", "^"])
    Node.add_symbols(symbols)
    s2t = {s["symbol"]: s for s in symbols}
    tree = tokens_to_tree(["A", "+", "C"], s2t)
    tree2 = tokens_to_tree(["A", "*", "A"], s2t)
    X = np.array([[0.1, 0.2],[0.2, 0.3],[0.3, 0.4],[0.4, 0.5]])
    fe = FastEval(X, 2, ["add", "sub", "mul", "div", "sin", "cos"])

    print(fe.execute(tree2))
    print(fe.execute(tree))


