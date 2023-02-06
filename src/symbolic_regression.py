import argparse
import json
import random
import time

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.termination.max_gen import MaximumGenerationTermination

from symbol_library import generate_symbol_library
from model import HVAE
from fasteval.fasteval import FastEval


def read_eq_data(filename):
    train = []
    with open(filename, "r") as file:
        for line in file:
            train.append([float(v) for v in line.strip().split(",")])
    return np.array(train)


def eval_vector(l, model, eval_obj):
    try:
        tree = model.decode(l)
        y_hat, constants = eval_obj.execute(tree)
        if not all(np.isfinite(y_hat)):
            error = 1e10
            # print("INF")
        else:
            error = np.sqrt(np.square(np.subtract(FastEval.X[:, -1], y_hat)).mean())
    except:
        print("Recursion limit")
        return 1e10, "", []
    return error, str(tree), constants


class SRProblem(ElementwiseProblem):
    def __init__(self, model, eval_object, dim):
        self.model = model
        self.eval_object = eval_object
        self.input_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        self.best_f = 9e+50
        self.best_expr = None
        self.models = dict()
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        error, expr, constants = eval_vector(torch.tensor(x[None, None, :]), self.model, self.eval_object)
        if expr in self.models:
            self.models[expr]["trees"] += 1
        else:
            constants = [float(c) for c in constants]
            self.models[expr] = {"expr": expr, "error": error, "trees": 1, "const": constants}
            if error < self.best_f:
                self.best_f = error
                self.best_expr = self.models[expr]
                print(f"New best expression: {expr}, with constants [{','.join([str(c) for c in constants])}]")
        out["F"] = error


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
        new = []
        for i in range(X.shape[0]):
            eq = problem.model.decode(torch.tensor(X[i, :])[None, None, :])
            var = problem.model.encode(eq)[1][0, 0].detach().numpy()
            mutation_scale = np.random.random()
            std = mutation_scale * (np.exp(var / 2.0) - 1) + 1
            new.append(torch.normal(torch.tensor(mutation_scale*X[i]), std=torch.tensor(std)).numpy())
        return np.array(new, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Nguyen benchmark', description='Run a ED benchmark')
    parser.add_argument("-dataset", required=True)
    parser.add_argument("-baseline", choices=['HVAE_evo'], required=True)
    parser.add_argument("-symbols", nargs="+", required=True)
    parser.add_argument("-num_vars", default=2, type=int)
    parser.add_argument("-has_const", action="store_true")
    parser.add_argument("-latent", default=32, type=int)
    parser.add_argument("-params", required=True)
    parser.add_argument("-success_threshold", default=1e-8)
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()


    # -----------------------------------------------------------------------------------------------------------------
    #
    #                           WORK IN PROGRESS, USE SR SCRIPTS FROM ProGED
    #           (https://github.com/smeznar/ProGED/blob/main/ProGED/examples/ng_bench.py)
    #                                     TO EVALUATE THE RESULTS
    #
    # -----------------------------------------------------------------------------------------------------------------

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Read data
    train = read_eq_data(args.dataset)
    symbols = generate_symbol_library(args.num_vars, args.symbols, args.has_const)
    input_dim = len(symbols)
    HVAE.add_symbols(symbols)
    model = torch.load(args.params)
    fe = FastEval(train, args.num_vars, symbols, has_const=args.has_const)

    if args.baseline == "HVAE_evo":
        ga = GA(pop_size=200, sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(model, fe, args.latent)
        res = minimize(problem, ga, BestTermination(min_f=args.success_threshold), verbose=True)
        with open(f"../results/nguyen/{args.dataset.strip().split('/')[-1]}_{time.time()}.json", "w") as file:
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
