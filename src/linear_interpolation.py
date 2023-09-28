import torch

from model import HVAE
from utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library


def interpolateAB(model, treeA, treeB, steps=5):
    treeBA = create_batch([treeA])
    treeBB = create_batch([treeB])
    l1 = model.encode(treeBA)[0]
    l2 = model.encode(treeBB)[0]
    print(f"Expr A:\t{str(treeA)}")
    print(f"a=0:\t{str(model.decode(l1)[0])}")
    for i in range(1, steps-1):
        a = i/(steps-1)
        la = (1-a) * l1 + a * l2
        print(f"a={str(a)[:5]}:\t{str(model.decode(la)[0])}")
    print(f"a=1:\t{str(model.decode(l2)[0])}")
    print(f"Expr B:\t{str(treeB)}")


if __name__ == '__main__':
    config = load_config_file("../configs/test_config.json")
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)

    param_file = "../params/ng1_7.pt"
    model = torch.load(param_file)

    exprA = "cos ( A * A + A ) + exp ( A ) / A"
    exprB = "A ^2 - A ^3"
    steps = 5

    tokensA = exprA.split(" ")
    tokensB = exprB.split(" ")
    treeA = tokens_to_tree(tokensA, so)
    treeB = tokens_to_tree(tokensB, so)

    interpolateAB(model, treeA, treeB)
