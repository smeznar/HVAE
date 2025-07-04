import torch
from SRToolkit.utils import tokens_to_tree, generate_n_expressions
from SRToolkit.dataset import SRBenchmark

from model import HVAE
from train import create_batch


def interpolateAB(model, treeA, treeB, symbol2index, steps=5):
    treeBA = create_batch([treeA], symbol2index)
    treeBB = create_batch([treeB], symbol2index)
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
    dataset = SRBenchmark.feynman("../data/fey_data").create_dataset("I.29.4")
    symbol2index = dataset.symbols.symbols2index()

    model_parameters = "../params/24random.pt"
    model = HVAE(len(dataset.symbols), 24, dataset.symbols)
    model.load_state_dict(torch.load(model_parameters, weights_only=True))
    model.eval()

    # Expressions we want to interpolate between
    expressions = generate_n_expressions(dataset.symbols, 2, max_expression_length=30)
    treeA = tokens_to_tree(expressions[0], dataset.symbols)
    treeB = tokens_to_tree(expressions[1], dataset.symbols)

    # Number of steps in the interpolation (inclusive with expressions A and B)
    steps = 5
    interpolateAB(model, treeA, treeB, symbol2index)
