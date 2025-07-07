import torch
from SRToolkit.utils import tokens_to_tree, generate_n_expressions
from SRToolkit.dataset import SRBenchmark
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm
import numpy as np

from EDHiE.model import BatchedNode, HVAE

def create_batch(trees, symbol2index):
    t = BatchedNode(symbol2index, trees=trees)
    t.create_target()
    return t

class TreeBatchSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq
        self.permute = np.random.permutation(self.num_eq)

    def __iter__(self):
        for i in range(len(self)):
            batch = self.permute[(i*self.batch_size):((i+1)*self.batch_size)]
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


def logistic_function(iter, total_iters, supremum=0.04):
    x = iter/total_iters
    return supremum/(1+50*np.exp(-10*x))


def train_hvae(model, trainset, symbol_library, epochs=20, batch_size=32, verbose=True):
    symbol2index = symbol_library.symbols2index()
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    total_iters = epochs*(len(trainset)//batch_size)
    lmbda = logistic_function(iter_counter, total_iters)

    midpoint = len(trainset) // (2 * batch_size)

    for epoch in range(epochs):
        sampler = TreeBatchSampler(batch_size, len(trainset))
        bce, kl, total, num_iters = 0, 0, 0, 0

        with tqdm(total=len(trainset), desc=f'Training HVAE - Epoch: {epoch + 1}/{epochs}', unit='chunks') as prog_bar:
            for i, tree_ids in enumerate(sampler):
                batch = create_batch([trainset[j] for j in tree_ids], symbol2index)

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
                    for i in range(1):
                        print()
                        print(f"O: {"".join(original_trees[i].to_list(symbol_library=symbol_library))}")
                        print(f"P: {"".join(decoded_trees[i].to_list(symbol_library=symbol_library))}")


if __name__ == '__main__':
    dataset = SRBenchmark.feynman("../data/fey_data").create_dataset("I.29.4")
    latent_size = 24
    num_expressions = 30000
    max_expression_length = 30
    model_name = "24random"

    # Possibly create a training set or load expressions
    expressions = generate_n_expressions(dataset.symbols, num_expressions, max_expression_length=max_expression_length)
    expr_tree = [tokens_to_tree(expr, dataset.symbols) for expr in expressions]
    # Create a training set
    trainset = TreeDataset(expr_tree)

    # Train the model
    model = HVAE(len(dataset.symbols), latent_size, dataset.symbols)
    train_hvae(model, trainset, dataset.symbols, epochs=40)
    torch.save(model.state_dict(), f"../params/{model_name}.pt")