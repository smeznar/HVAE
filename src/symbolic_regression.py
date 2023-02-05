import argparse
import json

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


def read_eq_data(eq_number):
    train = []
    test = []
    with open(f"/home/sebastian/Downloads/Feynman_with_units/{eq_number}", "r") as file:
        for i, row in enumerate(file):
            line = [float(t) for t in row.strip().split(" ")]
            if i < 10000:
                train.append(line)
            elif i < 20000:
                test.append(line)
            else:
                break
    return np.array(train), np.array(test)


def eval_vector(l, model, eval_obj):
    tree = model.decode(l)
    y_hat, constants = eval_obj.execute(tree)
    error = np.sqrt(np.square(np.subtract(FastEval.X[:, -1], y_hat)).mean())
    return error, str(tree), constants


class SRProblem(ElementwiseProblem):
    def __init__(self, model, eval_object, dim):
        self.model = model
        self.eval_object = eval_object
        self.input_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        self.best_f = 9e+50
        self.models = dict()
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        error, expr, constants = eval_vector(torch.tensor(x[None, None, :]), self.model, self.eval_object)
        if expr in self.models:
            self.models[expr]["trees"] += 1
        else:
            self.models[expr] = {"expr": expr, "error": error, "trees": 1, "const": constants}
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
    parser.add_argument("-params")
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()

    # Read data
    train, test = read_eq_data(args.dataset)
    symbols = generate_symbol_library(args.num_vars, args.symbols, args.has_const)
    input_dim = len(symbols)
    HVAE.add_symbols(symbols)
    model = HVAE(input_dim, args.latent)
    fe = FastEval(train, args.num_vars, symbols, has_const=args.has_const)

    if args.baseline == "HVAE_evo":
        ga = GA(pop_size=200, sampling=TorchNormalSampling(), crossover=LICrossover(), mutation=RandomMutation(),
                eliminate_duplicates=False)
        problem = SRProblem(model, fe, args.latent)
        res = minimize(problem, ga, BestTermination(), verbose=True)
        # with open(f"results/hvae_evo/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json", "w") as file:
        #     for i in range(len(problem.models)):
        #         problem.models[i]["trees"] = problem.evaluated_models[problem.models[i]["eq"]]
        #     json.dump(problem.models, file)

    # if args.baseline == "HVAE_random":
    #     fe = FastEval(train, args.num_vars, symbols, has_const=args.has_const)
    #     generator = GeneratorHVAE(args.params, ["X"], universal_symbols)
    #     ed = EqDisco(data=train, variable_names=["X", 'Y'], generator=generator, sample_size=100000, verbosity=0)
    #     ed.generate_models()
    #     ed.fit_models()
    #     print(len(ed.models))
    #     print(ed.get_results())
    #     ed.write_results(f"results/hvae_random_{args.dimension}/nguyen_{args.eq_num}_{np.random.randint(0, 1000000)}.json")
