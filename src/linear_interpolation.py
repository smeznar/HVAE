import torch

from model import HVAE
from utils import tokens_to_tree
from symbol_library import generate_symbol_library


def interpolateAB(model, exprA, exprB, steps=5):
    tokensA = exprA.split(" ")
    tokensB = exprB.split(" ")
    treeA = tokens_to_tree(tokensA, s2t)
    treeB = tokens_to_tree(tokensB, s2t)

    l1 = model.encode(treeA)[0]
    l2 = model.encode(treeB)[0]
    print(f"Expr A:\t{str(treeA)}")
    print(f"a=0:\t{str(model.decode(l1))}")
    for i in range(1, steps-1):
        a = i/(steps-1)
        la = (1-a) * l1 + a * l2
        print(f"a={str(a)[:5]}:\t{str(model.decode(la))}")
    print(f"a=1:\t{str(model.decode(l2))}")
    print(f"Expr B:\t{str(treeB)}")


if __name__ == '__main__':
    param_file = "../params/4_2k.pt"
    symbols = generate_symbol_library(1, ["+", "-", "*", "/", "^"])
    HVAE.add_symbols(symbols)
    s2t = {s["symbol"]: s for s in symbols}
    steps = 5

    model = torch.load(param_file)

    interpolateAB(model, "A + A / A", "A * C ^ A")
    # TODO: Create reproducible results of linear interpolation and add them to the paper
