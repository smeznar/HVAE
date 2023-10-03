import json
import random

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
from hvae_utils import load_config_file, create_batch
from evaluation import RustEval

# -----------------------------------------------------------------------------------------------------------------
#
#                           WORK IN PROGRESS, USE SR SCRIPTS FROM ProGED
#           (https://github.com/smeznar/ProGED/blob/main/ProGED/examples/ng_bench.py)
#                                     TO EVALUATE THE RESULTS
#
# -----------------------------------------------------------------------------------------------------------------


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
            if error is None:
                self.models[expr_postfix_str] = {"expr": str(tree), "error": self.default_value, "trees": 1}
                error = self.default_value
            else:
                self.models[expr_postfix_str] = {"expr": str(tree), "error": error, "trees": 1, "constants": constants}

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
        mutation_scale = np.random.random((200, 1))
        std = np.multiply(mutation_scale, (np.exp(var/2.0)-1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)


if __name__ == '__main__':
    config = load_config_file("../configs/test_config.json")
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    reconstruction_config = config["reconstruction"]
    sr_config = config["symbolic_regression"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])
        random.seed(training_config["seed"])

    # Read data
    train = read_eq_data(sr_config["train_set_path"])
    fe_train = RustEval(train)

    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    HVAE.add_symbols(sy_lib)
    model = torch.load(training_config["param_path"])

    if sr_config["baseline"] == "EDHiE":
        ga = GA(pop_size=sr_config["population_size"], sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(model, fe_train, training_config["latent_size"])
        res = minimize(problem, ga, BestTermination(min_f=sr_config["success_threshold"], n_max_gen=sr_config["max_generations"]), verbose=True)
        with open(sr_config["results_path"], "w") as file:
            json.dump({"best": problem.best_expr, "all": list(problem.models.values())}, file)

    # if args.baseline == "HVAE_random":
    #     fe = FastEval(train, args.num_vars, symbols, has_const=args.has_const)
    #     generator = GeneratorHVAE(args.params, ["X"], universal_symbols)
    #     ed = EqDisco(data=train, variable_names=["X", 'Y'], generator=generator, sample_size=100000, verbosity=0)
    #     ed.generate_models()
    #     ed.fit_models()
    #     print(len(ed.models))
    #     print(ed.get_results())
    #     ed.write_results(f"results/hvae_random_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
