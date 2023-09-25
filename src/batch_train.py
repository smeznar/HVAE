from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from tqdm import tqdm

# from utils import tokens_to_tree, read_expressions
from utils import read_expressions_json
from batch_model import HVAE
from symbol_library import generate_symbol_library
from tree import BatchedNode


def collate_fn(batch):
    return batch


class TreeSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq

    def __iter__(self):
        for i in range(len(self)):
            batch = np.random.randint(low=0, high=self.num_eq, size=self.batch_size)
            yield batch

    def __len__(self):
        return self.num_eq // self.batch_size


class TreeBatchSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq
        self.permute = np.random.permutation(self.num_eq)

    def __iter__(self):
        for i in range(len(self)):
            batch = self.permute[(i*32):((i+1)*32)]
            yield batch

    def __len__(self):
        return self.num_eq // self.batch_size


class TreeDataset(Dataset):
    def __init__(self, train):
        self.train = train

    def __getitem__(self, idx):
        return self.train[idx]

    def __len__(self):
        return len(self.train)


def create_batch(trees):
    t = BatchedNode(trees=trees)
    t.create_target()
    return t


def logistic_function(iter, total_iters, supremum=0.045):
    x = iter/total_iters
    return supremum/(1+50*np.exp(-10*x))


def train_hvae(model, trees, epochs=20, batch_size=32, verbose=True):
    dataset = TreeDataset(trees)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    total_iters = epochs*(len(dataset)//batch_size)
    lmbda = logistic_function(iter_counter, total_iters)

    midpoint = len(dataset) // (2 * batch_size)

    for epoch in range(epochs):
        sampler = TreeBatchSampler(batch_size, len(dataset))
        bce, kl, total, num_iters = 0, 0, 0, 0

        with tqdm(total=len(dataset), desc=f'Testing - Epoch: {epoch + 1}/{epochs}', unit='chunks') as prog_bar:
            for i, tree_ids in enumerate(sampler):
                batch = create_batch([dataset[j] for j in tree_ids])

                mu, logvar, outputs = model(batch)
                loss, bcel, kll = outputs.loss(mu, logvar, lmbda, criterion)
                bce += bcel.detach().item()
                kl += kll.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_iters += 1
                prog_bar.set_postfix(**{'run:': "HVAE",
                                        'loss': (bce+kl) / num_iters,
                                        'BCE': bce / num_iters,
                                        'KLD': kl / num_iters})
                prog_bar.update(batch_size)

                lmbda = logistic_function(iter_counter, total_iters)
                iter_counter += 1

                if verbose and i == midpoint:
                    original_trees = batch.to_expr_list()
                    z = model.encode(batch)[0]
                    decoded_trees = model.decode(z)
                    for i in range(10):
                        print("--------------------")
                        print(f"O: {original_trees[i]}")
                        print(f"P: {decoded_trees[i]}")
                    a = 0


if __name__ == '__main__':
    parser = ArgumentParser(prog='Train HVAE', description='A script for training the HVAE model.')
    parser.add_argument("-expressions", required=True)
    parser.add_argument("-symbols", nargs="+", required=True)
    parser.add_argument("-batch", default=32, type=int)
    parser.add_argument("-num_vars", default=2, type=int)
    parser.add_argument("-has_const", action="store_true")
    parser.add_argument("-latent_size", default=32, type=int)
    parser.add_argument("-epochs", default=20, type=int)
    parser.add_argument("-param_path", default="")
    parser.add_argument("-annealing_iters", default=3000, type=int)
    parser.add_argument("-verbose", action="store_true")
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    symbols = generate_symbol_library(args.num_vars, args.symbols, args.has_const)
    HVAE.add_symbols(symbols)

    s2t = {s["symbol"]: s for s in symbols}
    trees = read_expressions_json(args.expressions)

    model = HVAE(len(symbols), args.latent_size)

    train_hvae(model, trees, args.epochs, args.batch, args.verbose)

    if args.param_path != "":
        torch.save(model, args.param_path)
