from typing import Union, List

import torch
import numpy as np
from SRToolkit.dataset import SR_dataset, SR_benchmark
from SRToolkit.utils import generate_n_expressions, tokens_to_tree

from EDHiE.train import TreeDataset, train_hvae
from EDHiE.model import HVAE
from EDHiE.symbolic_regression import symbolic_regression_run


def EDHiE(dataset: SR_dataset, grammar: Union[str, None]=None, size_trainset: int=50000, max_expression_length: int=35, trainset: Union[List[List[str]], None]=None,
          pretrained_params: Union[str, None]=None, latent_size: int=32, epochs: int=40, batch_size: int=32, max_height=10, save_params_to_file: Union[str, None]=None,
          num_runs: int=10, population_size: int=200, max_generations: int=500, seed: Union[int, None]=18, verbose: bool=True):

    if pretrained_params is None:
        # Generate trainset
        if trainset is None:
            if grammar is not None:
                trainset = generate_n_expressions(grammar, size_trainset, max_expression_length=max_expression_length, verbose=verbose)
            else:
                trainset = generate_n_expressions(dataset.symbols, size_trainset, max_expression_length=max_expression_length, verbose=verbose)

        expr_trees = [tokens_to_tree(expr, dataset.symbols) for expr in trainset]
        trainset = TreeDataset(expr_trees)
        # Train the HVAE model
        model = HVAE(len(dataset.symbols), latent_size, dataset.symbols, max_height)
        train_hvae(model, trainset, dataset.symbols, epochs, batch_size, verbose)
        if save_params_to_file is not None:
            torch.save(model.state_dict(), save_params_to_file)
        model.eval()
    else:
        model = HVAE(len(dataset.symbols), latent_size, dataset.symbols)
        model.load_state_dict(torch.load(pretrained_params, weights_only=True))
        model.eval()

    results = []
    for i in range(num_runs):
        if seed is not None:
            seed = seed + i
        else:
            seed = np.random.randint(1, np.iinfo(np.int64).max)

        if verbose:
            print()
            print("---------------------------------------------------------------------------")
            print(f"     Baseline: EDHiE, Run: {i+1}/{num_runs}")
            print("---------------------------------------------------------------------------")
            print()
            results.append(symbolic_regression_run(model, "EDHiE", dataset, seed, population_size, latent_size, max_generations, verbose))

    if verbose:
        print("Results EDHiE")
        num_successful = 0
        evaluated = []
        min_rmse = 1e20
        best_expression = None
        for i, result in enumerate(results):
            is_successful = result["min_rmse"] < dataset.success_threshold
            successful = "Successful" if is_successful else "Unsuccessful"
            evaluated_expressions = f"Number of evaluated expressions: {result['num_evaluated']}, " if is_successful else ""
            print(f"Run {i+1}/{len(results)}: {successful}; minimum RMSE: {result['min_rmse']}, {evaluated_expressions}best_expression: {result['best_expr']}")
            if result["min_rmse"] < min_rmse:
                min_rmse = result["min_rmse"]
                best_expression = result["best_expr"]
            if is_successful:
                num_successful += 1
                evaluated.append(result["num_evaluated"])
        if len(evaluated) > 0:
            evaluated_text = f", Average number of evaluated expressions {np.mean(evaluated)}(+- {np.std(evaluated)})"
        else:
            evaluated_text = ""
        print(f"Overall: Successful {num_successful}/{len(results)}, Best expressions found {''.join(best_expression)} had RMSE {min_rmse}{evaluated_text}")
    return results


def HVAR(dataset: SR_dataset, grammar: Union[str, None]=None, size_trainset: int=50000, max_expression_length: int=35, trainset: Union[List[List[str]], None]=None,
          pretrained_params: Union[str, None]=None, latent_size: int=32, epochs: int=40, batch_size: int=32, save_params_to_file: Union[str, None]=None,
          num_runs: int=10, expr_generated: int=100000, seed: Union[int, None]=18, verbose: bool=True):

    if pretrained_params is None:
        # Generate trainset
        if trainset is None:
            if grammar is not None:
                trainset = generate_n_expressions(grammar, size_trainset, max_expression_length=max_expression_length, verbose=verbose)
            else:
                trainset = generate_n_expressions(dataset.symbols, size_trainset, max_expression_length=max_expression_length, verbose=verbose)

        expr_trees = [tokens_to_tree(expr, dataset.symbols) for expr in trainset]
        trainset = TreeDataset(expr_trees)
        # Train the HVAE model
        model = HVAE(len(dataset.symbols), latent_size, dataset.symbols)
        train_hvae(model, trainset, dataset.symbols, epochs, batch_size, verbose)
        if save_params_to_file is not None:
            torch.save(model.state_dict(), save_params_to_file)
        model.eval()
    else:
        model = HVAE(len(dataset.symbols), latent_size, dataset.symbols)
        model.load_state_dict(torch.load(pretrained_params, weights_only=True))
        model.eval()

    results = []
    for i in range(num_runs):
        if seed is not None:
            seed = seed + i
        else:
            seed = np.random.randint(np.iinfo(np.int64).max)

        if verbose:
            print()
            print("---------------------------------------------------------------------------")
            print(f"     Baseline: HVAR, Run: {i+1}/{num_runs}")
            print("---------------------------------------------------------------------------")
            print()
            results.append(symbolic_regression_run(model, "HVAR", dataset, seed, 1, latent_size, expr_generated, verbose))

    if verbose:
        print("Results HVAR")
        num_successful = 0
        evaluated = []
        min_rmse = 1e20
        best_expression = None
        for i, result in enumerate(results):
            is_successful = result["min_rmse"] < dataset.success_threshold
            successful = "Successful" if is_successful else "Unsuccessful"
            evaluated_expressions = f"Number of evaluated expressions: {result['num_evaluated']}, " if is_successful else ""
            print(f"Run {i+1}/{len(results)}: {successful}; minimum RMSE: {result['min_rmse']}, {evaluated_expressions}best_expression: {result['best_expr']}")
            if result["min_rmse"] < min_rmse:
                min_rmse = result["min_rmse"]
                best_expression = result["best_expr"]
            if is_successful:
                num_successful += 1
                evaluated.append(result["num_evaluated"])
        if len(evaluated) > 0:
            evaluated_text = f", Average number of evaluated expressions {np.mean(evaluated)}(+- {np.std(evaluated)})"
        else:
            evaluated_text = ""
        print(f"Overall: Successful {num_successful}/{len(results)}, Best expressions found {''.join(best_expression)} had RMSE {min_rmse}{evaluated_text}")
    return results


if __name__ == '__main__':
    grammar = """E -> E '+' F [0.2004]
E -> E '-' F [0.1108]
E -> F [0.6888]
F -> F '*' T [0.3349]
F -> F '/' T [0.1098]
F -> T [0.5553]
T -> 'C' [0.1174]
T -> R [0.1746]
T -> V [0.708]
R -> '(' E ')' [0.6841]
R -> E '^2' [0.00234]
R -> E '^3' [0.00126]
R -> 'sin' '(' E ')' [0.028]
R -> 'cos' '(' E ')' [0.049]
R -> 'sqrt' '(' E ')' [0.0936]
R -> 'exp' '(' E ')' [0.0878]
R -> 'ln' '(' E ')' [0.0539]
V -> 'X_0' [0.33]
V -> 'X_1' [0.33]
V -> 'X_2' [0.34]
"""

    benchmark = SR_benchmark.feynman("../data/fey_data/")
    dataset = benchmark.create_dataset("I.12.4")
    EDHiE(dataset, size_trainset=10000, epochs=20, latent_size=24, num_runs=10, verbose=True,
          max_generations=250, population_size=40, max_expression_length=30, seed=None)