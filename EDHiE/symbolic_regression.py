import random

import numpy as np
import torch
from SRToolkit.dataset import SRBenchmark, SRDataset
from SRToolkit.evaluation import SR_evaluator
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.termination.max_gen import MaximumGenerationTermination

# import nevergrad as ng

from model import HVAE
from train import create_batch


class SRProblem(Problem):
    def __init__(self, model, evaluator: SR_evaluator, dim, default_value=1e10, verbose=True):
        self.model = model
        self.default_value = np.array(default_value)
        self.evaluator = evaluator
        self.symbol2index = self.evaluator.symbol_library.symbols2index()
        self.input_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        self.best_f = 9e+50
        self.models = dict()
        self.verbose = verbose
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        trees = self.model.decode(torch.tensor(x[:, :]))

        errors = []
        for tree in trees:
            expr = tree.to_list("infix", self.evaluator.symbol_library)
            error = self.evaluator.evaluate_expr(expr)
            if error < self.best_f:
                self.best_f = error
                if self.verbose:
                    print(f"New best expression with score {error}: {''.join(expr)}")
            errors.append(error)

        out["F"] = np.array(errors)


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
        return (X[0, :] * weights[:, None] + X[1, :] * (1 - weights[:, None]))[None, :, :]


class RandomMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        trees = problem.model.decode(torch.tensor(X))
        batch = create_batch(trees, problem.symbol2index)
        var = problem.model.encode(batch)[1].detach().numpy()
        mutation_scale = np.random.random((X.shape[0], 1))
        std = np.multiply(mutation_scale, (np.exp(var/2.0)-1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)


def symbolic_regression_run(model, approach, dataset: SRDataset, seed, population_size=40, latent_size=24, max_generations=250, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    evaluator = dataset.create_evaluator()
    if approach == "EDHiE":
        ga = GA(pop_size=population_size, sampling=TorchNormalSampling(), crossover=LICrossover(),
                mutation=RandomMutation(), eliminate_duplicates=False)
        problem = SRProblem(model, evaluator, latent_size, verbose)
        minimize(problem, ga, BestTermination(min_f=dataset.success_threshold, n_max_gen=max_generations), verbose=verbose)

        return evaluator.get_results(top_k=-1, success_threshold=dataset.success_threshold)

    elif approach == "HVAR":
        gaussian_distribution_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        min_score = 1e20
        for _ in range(population_size*max_generations):
            x = torch.normal(gaussian_distribution_mean)[None, :]
            tree = model.decode(x)[0]
            expr = tree.to_list("infix", dataset.symbols)
            expr_score = evaluator.evaluate_expr(expr)
            if expr_score < min_score and verbose:
                min_score = expr_score
                print(f"New best expression with score {min_score}: {''.join(expr)}")
                if min_score < dataset.success_threshold:
                    break
        return evaluator.get_results()

    else:
        print(f"No approach with the name {approach}! Check spelling.")


    # elif baseline == "nevergrad":
    #     models = nevergrad_sr(model, re_train, training_config["latent_size"],
    #                           budget=sr_config["population_size"]*sr_config["max_generations"],
    #                           success_threshold=sr_config["success_threshold"],
    #                           default_value=sr_config["default_error"]).models
    #     best_candidates = sorted(list(models.values()), key=lambda x: x['error'])
    #     if sr_config["save_best_n"] > -1:
    #         best_candidates = best_candidates[:sr_config["save_best_n"]]
    #     return {"baseline": baseline,
    #             "train": {"best_expr": best_candidates[0]['expr'], "best_error": best_candidates[0]["error"]},
    #             "all_evaluated": len(models), "all_generated": sum([m["trees"] for m in models.values()]),
    #             "best_candidates": best_candidates}

# class Nevergrad_eval:
#     def __init__(self, model, eval_object, default_value=1e20):
#         self.models = {}
#         self.model = model
#         self.eval_object = eval_object
#         self.default_value = default_value
#         self.best_f = 9e50
#         self.best_expr = ""
#
#     def eval_vec(self, vec):
#         tree = model.decode(torch.tensor(vec[None, None, :], dtype=torch.float32))[0]
#         expr_postfix = tree.to_list(notation="postfix")
#         expr_postfix_str = "".join(expr_postfix)
#         if expr_postfix_str in self.models:
#             self.models[expr_postfix_str]["trees"] += 1
#             return self.models[expr_postfix_str]["error"]
#         else:
#             error, constants = self.eval_object.fit_and_evaluate(expr_postfix)
#             expr_infix = " ".join(tree.to_list())
#             if error is None or error > self.default_value:
#                 self.models[expr_postfix_str] = {"expr": expr_infix, "error": self.default_value, "constants": [], "trees": 1}
#                 error = self.default_value
#             else:
#                 self.models[expr_postfix_str] = {"expr": expr_infix, "error": error, "trees": 1, "constants": constants}
#
#             if error < self.best_f:
#                 self.best_f = error
#                 self.best_expr = str(tree)
#                 print()
#                 print(f"New best expression: {self.best_expr}, with constants [{','.join([str(c) for c in constants])}]"
#                       f"\t|\tError: {self.best_f}")
#             return error
#
#
# def nevergrad_sr(model, eval_object, model_size, budget=100000, success_threshold=1e-6, default_value=1e20):
#     ne = Nevergrad_eval(model, eval_object, default_value)
#     parametrization = ng.p.Array(shape=(model_size,), lower=-3.5, upper=3.5)
#     optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget, num_workers=1)
#     early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.current_bests["minimum"].mean < success_threshold)
#     optimizer.register_callback("ask", early_stopping)
#     optimizer.register_callback("tell", ng.callbacks.ProgressBar())
#     values = optimizer.minimize(ne.eval_vec)
#     return ne

if __name__ == '__main__':
    # Load the ED/SR dataset
    # SRBenchmark.feynman("../data/fey_data").list_datasets(num_variables=2)
    dataset = SRBenchmark.feynman("../data/fey_data").create_dataset("II.11.28")

    # Load the HVAE model
    latent_size = 24
    model_parameters = "../params/24random.pt"
    model = HVAE(len(dataset.symbols), latent_size, dataset.symbols)
    model.load_state_dict(torch.load(model_parameters, weights_only=True))
    model.eval()

    # Run symbolic regression
    approaches = ["EDHiE"]
    number_of_runs = 10
    seed = 18
    population_size = 40
    max_generations = 250
    verbose = True
    results = []
    for approach in approaches:
        for i in range(number_of_runs):
            if seed is not None:
                seed = seed + i
            else:
                seed = np.random.randint(np.iinfo(np.int64).max)
            print()
            print("---------------------------------------------------------------------------")
            print(f"     Baseline: {approach}, Run: {i+1}/{number_of_runs}")
            print("---------------------------------------------------------------------------")
            print()
            results.append(symbolic_regression_run(model, approach, dataset, seed, population_size, latent_size, max_generations, verbose))

    # print(results)
