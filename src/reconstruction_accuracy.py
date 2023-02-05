from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import KFold
import editdistance

from utils import read_expressions, tokens_to_tree
from symbol_library import generate_symbol_library
from model import HVAE
from train import train_hvae


def one_fold(model, train, test, epochs, batch_size, annealing_iters, verbose):
    train_hvae(model, train, epochs, batch_size, annealing_iters, verbose)

    total_distance = []
    for t in test:
        latent = model.encode(t)[0]
        pt = model.decode(latent)
        total_distance.append(editdistance.eval(t.to_list(notation="postfix"), pt.to_list(notation="postfix")))

    return total_distance


def one_experiment(name, trees, input_dim, latent_dim, epochs, batch_size, annealing_iters, verbose, seed,
                   smaller_dataset=False, examples=2000):
    kf = KFold()
    distances = []
    for i, (train_idx, test_idx) in enumerate(kf.split(trees)):
        print(f"Fold {i + 1}")
        if smaller_dataset:
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            inds = np.random.permutation(train_idx)
            inds = inds[:examples]
            train = [trees[i] for i in inds]
        else:
            train = [trees[i] for i in train_idx]

        test = [trees[i] for i in test_idx]
        model = HVAE(input_dim, latent_dim)
        distances.append(one_fold(model, train, test, epochs, batch_size, annealing_iters, verbose))
        print(f"Mean: {np.mean(distances[-1])}, Var: {np.var(distances[-1])}")
        print()
    fm = [np.mean(d) for d in distances]
    with open("../results/hvae.txt", "a") as file:
        file.write(f"{name}\t Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}\n")
    print(f"Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}")
    return fm


if __name__ == '__main__':
    parser = ArgumentParser(prog='Train HVAE', description='A script for training the HVAE model.')
    parser.add_argument("-expressions", required=True)
    parser.add_argument("-symbols", nargs="+", required=True)
    parser.add_argument("-batch", default=32, type=int)
    parser.add_argument("-num_vars", default=2, type=int)
    parser.add_argument("-has_const", action="store_true")
    parser.add_argument("-latent_size", default=128, type=int)
    parser.add_argument("-epochs", default=20, type=int)
    parser.add_argument("-annealing_iters", default=1800, type=int)
    parser.add_argument("-verbose", action="store_true")
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    equations = read_expressions(args.expressions)
    symbols = generate_symbol_library(args.num_vars, args.symbols, args.has_const)
    input_dim = len(symbols)
    HVAE.add_symbols(symbols)

    s2t = {s["symbol"]: s for s in symbols}
    trees = [tokens_to_tree(eq, s2t) for eq in equations]

    one_experiment(args.expressions, trees, input_dim, args.latent_size, args.epochs, args.batch, args.annealing_iters,
                   args.verbose, args.seed)
