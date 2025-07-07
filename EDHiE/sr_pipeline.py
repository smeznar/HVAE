import torch
import numpy as np
from SRToolkit.dataset import SRDataset, SRBenchmark
from SRToolkit.utils import generate_n_expressions, tokens_to_tree


from train import TreeDataset, train_hvae
from model import HVAE
from symbolic_regression import symbolic_regression_run


def EDHiE(dataset: SRDataset, grammar=None, size_trainset=50000, max_expression_length=40, trainset=None,
          pretrain_params=None, latent_size=32, epochs=40, batch_size=32, save_params_to_file=None,
          num_runs=10, population_size=200, max_generations=500, seed=18, verbose=True):

    if pretrain_params is None:
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
        model.load_state_dict(torch.load(pretrain_params, weights_only=True))
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
            print(f"Run {i+1}/{len(result)}: {successful}; minimum RMSE: {result['min_rmse']}, {evaluated_expressions}best_expression: {result['best_expr']}")
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


def HVAR(dataset: SRDataset, grammar=None, size_trainset=50000, max_expression_length=35, trainset=None,
         pretrain_params=None, latent_size=32, epochs=40, batch_size=32, save_params_to_file=None,
         num_runs=10, expr_generated=100000, seed=18, verbose=True):

    if pretrain_params is None:
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
        model.load_state_dict(torch.load(pretrain_params, weights_only=True))
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
            print(f"Run {i+1}/{len(result)}: {successful}; minimum RMSE: {result['min_rmse']}, {evaluated_expressions}best_expression: {result['best_expr']}")
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
    dataset = SRBenchmark.feynman("../data/fey_data/").list_datasets(num_variables=2)
    # EDHiE(dataset, size_trainset=10000, epochs=20)