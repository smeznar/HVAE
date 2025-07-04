import numpy as np
import torch
from sklearn.model_selection import KFold
import editdistance
from SRToolkit.utils import generate_n_expressions, SymbolLibrary, tokens_to_tree

from train import train_hvae, TreeDataset, create_batch
from model import HVAE


def one_fold(model, train, test, symbol_library, epochs, batch_size, verbose):
    symbol2index = symbol_library.symbols2index()
    train_hvae(model, train, symbol_library, epochs=epochs, verbose=verbose)

    total_distance = []
    for i in range((len(test) // batch_size) + 1):
        batch = create_batch(test[(i*batch_size):((i+1)*batch_size)], symbol2index)
        latent = model.encode(batch)[0]
        exprs = model.decode(latent)
        for j in range(len(exprs)):
            total_distance.append(editdistance.eval(test[i*batch_size+j].to_list(notation="postfix"), exprs[j].to_list(notation="postfix")))
    return total_distance


def one_experiment(trees, symbol_library, input_dim, latent_dim, epochs, batch_size,
                   seed=None, train_examples=None, n_splits=5, verbose=True):
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int64).max-n_splits)

    kf = KFold(n_splits=n_splits)
    distances = []
    for i, (train_idx, test_idx) in enumerate(kf.split(trees)):
        print(f"Fold {i + 1}")
        if train_examples is not None:
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            inds = np.random.permutation(train_idx)
            inds = inds[:train_examples]
            train = [trees[i] for i in inds]
        else:
            train = [trees[i] for i in train_idx]

        train_set = TreeDataset(train)
        test_set = TreeDataset([trees[i] for i in test_idx])
        model = HVAE(input_dim, latent_dim, symbol_library)
        distances.append(one_fold(model, train_set, test_set, symbol_library, epochs, batch_size, verbose))
        print(f"Mean: {np.mean(distances[-1])}, Var: {np.var(distances[-1])}")
        print()
    return distances


if __name__ == '__main__':
    seed = 18
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Dataset to reconstruct
    num_exprs = 50000
    symbols_list = ["+", "-", "*", "/", "sin", "sqrt", "^2", "pi", "C"]
    num_variables = 2
    max_expr_length = 40

    symbols = SymbolLibrary.from_symbol_list(symbols_list, num_variables)
    expressions = generate_n_expressions(symbols, num_exprs, max_expression_length=max_expr_length)
    trees = [tokens_to_tree(expr, symbols) for expr in expressions]

    latent_dim = 32
    epochs = 40
    batch_size = 32
    train_examples = None
    n_splits = 5
    verbose = True

    distances = one_experiment(trees, symbols, len(symbols), latent_dim, epochs, batch_size, seed, train_examples, n_splits, verbose)

    # Write results to file
    dataset_name = f"random_set_{'_'.join(symbols_list)}_vars_{num_exprs}_max_length_{max_expr_length}"
    aggregated_results = [np.mean(d) for d in distances]
    print(f"Mean: {np.mean(aggregated_results)}, Std dev: {np.std(aggregated_results)}")
    if dataset_name is not None:
        with open(f"../results/reconsturction/{dataset_name}.json", "w") as file:
            file.write(f"{dataset_name}\t Mean: {np.mean(aggregated_results)}, Std dev: {np.std(aggregated_results)},"
                       f" All: {', '.join([str(f) for f in aggregated_results])}\n")


