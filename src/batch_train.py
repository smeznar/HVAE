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


# class TreeBatchSampler(Sampler):
#     def __init__(self, batch_size, num_eq):
#         self.batch_size = batch_size
#         self.num_eq = num_eq
#
#     def __iter__(self):
#         for i in range(len(self)):
#             batch = np.random.randint(low=0, high=self.num_eq, size=self.batch_size)
#             BatchedNode(
#             yield batch
#
#     def __len__(self):
#         return self.num_eq // self.batch_size


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

def train_hvae(model, trees, epochs=20, batch_size=32, annealing_iters=2800, verbose=True):
    dataset = TreeDataset(trees)
    sampler = TreeSampler(batch_size, len(dataset))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    lmbda = (np.tanh(-4.5) + 1) / 2

    midpoint = len(dataset) // (8 * batch_size)

    for epoch in range(epochs):
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

                if iter_counter < annealing_iters:
                    lmbda = (np.tanh((iter_counter - 4500) / 1000) + 1) / 2
                iter_counter += 1

                if verbose and i == midpoint:
                    z = model.encode(batch)[0]
                    decoded_tree = model.decode(z)
                    a = 0
                #     print("\nO: {}".format(str(trees[0])))
                #     print("P: {}".format(str(decoded_tree)))
                #


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
    parser.add_argument("-annealing_iters", default=2800, type=int)
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

    train_hvae(model, trees, args.epochs, args.batch, args.annealing_iters, args.verbose)

    if args.param_path != "":
        torch.save(model, args.param_path)
