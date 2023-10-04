import json
import random
from argparse import ArgumentParser

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.termination.max_gen import MaximumGenerationTermination

from symbol_library import generate_symbol_library
from model import HVAE
from hvae_utils import load_config_file, create_batch, tokens_to_tree
from evaluation import RustEval


def read_eq_data(filename):
    train = []
    with open(filename, "r") as file:
        for line in file:
            train.append([float(v) for v in line.strip().split(",")])

    return np.array(train)


class SRProblem(Problem):
    def __init__(self, model, eval_object, dim, default_value=1e10):
        self.model = model
        self.default_value = default_value
        self.eval_object = eval_object
        self.input_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        self.best_f = 9e+50
        self.best_expr = None
        self.models = dict()
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        trees = self.model.decode(torch.tensor(x[:, None, :]))

        errors = []
        for tree in trees:
            error = self.eval_expression(tree)
            errors.append(error)

        out["F"] = np.array(errors)

    def eval_expression(self, tree):
        expr_postfix = tree.to_list(notation="postfix")
        expr_postfix_str = "".join(expr_postfix)
        if expr_postfix_str in self.models:
            self.models[expr_postfix_str]["trees"] += 1
            return self.models[expr_postfix_str]["error"]
        else:
            error, constants = self.eval_object.fit_and_evaluate(expr_postfix)
            expr_infix = " ".join(tree.to_list())
            if error is None:
                self.models[expr_postfix_str] = {"expr": expr_infix, "error": self.default_value, "trees": 1}
                error = self.default_value
            else:
                self.models[expr_postfix_str] = {"expr": expr_infix, "error": error, "trees": 1, "constants": constants}

            if error < self.best_f:
                self.best_f = error
                self.best_expr = str(tree)
                print(f"New best expression: {self.best_expr}, with constants [{','.join([str(c) for c in constants])}]"
                      f"\t|\tError: {self.best_f}")
            return error


class TorchNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [torch.normal(problem.input_mean).numpy() for _ in range(n_samples)]


class BestTermination(Termination):
    def __init__(self, min_f=1e-10, n_max_gen=500) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)

    def _update(self, algorithm):
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        return self.max_gen.update(algorithm)


class LICrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        weights = np.random.random(X.shape[1])
        return (X[0, :]*weights[:, None] + X[1, :]*(1-weights[:, None]))[None, :, :]


class RandomMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        trees = problem.model.decode(torch.tensor(X[:, None, :]))
        batch = create_batch(trees)
        var = problem.model.encode(batch)[1][:, 0, :].detach().numpy()
        mutation_scale = np.random.random((X.shape[0], 1))
        std = np.multiply(mutation_scale, (np.exp(var/2.0)-1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)


def one_sr_run(config, baseline, re_train, seed):
    training_config = config["training"]
    sr_config = config["symbolic_regression"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if baseline == "EDHiE":
        ga = GA(pop_size=sr_config["population_size"], sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(model, re_train, training_config["latent_size"], default_value=sr_config["default_error"])
        minimize(problem, ga, BestTermination(min_f=sr_config["success_threshold"], n_max_gen=sr_config["max_generations"]), verbose=True)

        best_candidates = sorted(list(problem.models.values()), key=lambda x: x['error'])
        if sr_config["save_best_n"] > -1:
            best_candidates = best_candidates[:sr_config["save_best_n"]]
        return {"baseline": baseline, "train": {"best_expr": problem.best_expr, "best_error": problem.best_f},
                "all_evaluated": len(problem.models), "all_generated": sum([m["trees"] for m in problem.models.values()]),
                "best_candidates": best_candidates}

    elif baseline == "HVAR":
        gaussian_distribution_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        best_f = 9e+50
        best_expr = ""
        models = {}
        for _ in range(sr_config["population_size"]*sr_config["max_generations"]):
            x = torch.normal(gaussian_distribution_mean)[None, None, :]
            tree = model.decode(x)[0]
            expr_postfix = tree.to_list(notation="postfix")
            expr_postfix_str = "".join(expr_postfix)
            if expr_postfix_str in models:
                models[expr_postfix_str]["trees"] += 1
            else:
                error, constants = re_train.fit_and_evaluate(expr_postfix)
                expr_infix = " ".join(tree.to_list())
                if error is None:
                    models[expr_postfix_str] = {"expr": expr_infix, "error": sr_config["default_error"], "trees": 1}
                else:
                    models[expr_postfix_str] = {"expr": expr_infix, "error": error, "trees": 1, "constants": constants}
                    if error < best_f:
                        best_f = error
                        best_expr = str(tree)
                        print(f"New best expression: {best_expr}, with constants [{','.join([str(c) for c in constants])}]"
                            f"\t|\tError: {best_f}")
                    if error < sr_config["success_threshold"]:
                        break

        best_candidates = sorted(list(models.values()), key=lambda x: x['error'])
        if sr_config["save_best_n"] > -1:
            best_candidates = best_candidates[:sr_config["save_best_n"]]
        return {"baseline": baseline, "train": {"best_expr": best_candidates[0]['expr'], "best_error": best_candidates[0]["error"]},
                "all_evaluated": len(models), "all_generated": sum([m["trees"] for m in models.values()]),
                "best_candidates": best_candidates}


def check_on_test_set(results, re_test, so):
    best_test_error = 9e+50
    best_test_expression = ""

    for i in range(len(results["best_candidates"])):
        tree = tokens_to_tree(results["best_candidates"][i]["expr"].split(" "), so)
        test_error = re_test.get_error(tree.to_list("postfix"), [results["best_candidates"][i]["constants"]])[0]
        results["best_candidates"][i]["test_error"] = test_error
        if test_error < best_test_error:
            best_test_error = test_error
            best_test_expression = str(tree)

    test_best = {}
    test_best["best_error"] = best_test_error
    test_best["best_expr"] = best_test_expression
    results["test"] = test_best
    return results


if __name__ == '__main__':
    parser = ArgumentParser(prog='Symbolic regression', description='Run a symbolic regression benchmark')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    sr_config = config["symbolic_regression"]
    expr_config = config["expression_definition"]
    training_config = config["training"]

    train_set = read_eq_data(sr_config["train_set_path"])
    re_train = RustEval(train_set, default_value=sr_config["default_error"])

    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)
    model = torch.load(training_config["param_path"])

    results = []
    for baseline in sr_config["baselines"]:
        for i in range(sr_config["number_of_runs"]):
            if sr_config["seed"] is not None:
                seed = sr_config["seed"] + i
            else:
                seed = np.random.randint(np.iinfo(np.int64).max)
            print()
            print("---------------------------------------------------------------------------")
            print(f"     Baseline: {baseline}, Run: {i+1}/{sr_config['number_of_runs']}")
            print("---------------------------------------------------------------------------")
            print()
            results.append(one_sr_run(config, baseline, re_train, seed))

    test_set = read_eq_data(sr_config["test_set_path"])
    re_test = RustEval(train_set, default_value=sr_config["default_error"])
    for i in range(len(results)):
        results[i] = check_on_test_set(results[i], re_test, so)

    with open(sr_config["results_path"], "w") as file:
        json.dump(results, file)
