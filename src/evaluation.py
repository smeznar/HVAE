from rusteval import Evaluator

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination


def read_eq_data(filename):
    train = []
    with open(filename, "r") as file:
        for line in file:
            train.append([float(v) for v in line.strip().split(",")])
    return np.array(train)


class PymooProblem(ElementwiseProblem):
    def __init__(self, model, constants, evaluator, estimation_settings, lower=-5, upper=5, default_error=1e10):
        xl = np.full(constants, lower)
        xu = np.full(constants, upper)
        super().__init__(n_var=constants, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.model = model
        self.evaluator = evaluator
        self.default_error = default_error
        self.estimation_settings = estimation_settings

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            rmse = self.evaluator.get_rmse(self.model, [float(v) for v in x])
            out["F"] = rmse
        except:
            out["F"] = self.default_error


def DE_pymoo(model, constants, evaluator, **estimation_settings):
    pymoo_problem = PymooProblem(model, constants, evaluator, estimation_settings)
    strategy = "DE/best/1/bin"
    algorithm = DE(
        pop_size=20,
        sampling=LHS(),
        variant=strategy,
        CR=0.7,
        dither="vector",
        jitter=False
    )

    termination = DefaultSingleObjectiveTermination(
        xtol=0.7,
        cvtol=1e-6,
        ftol=1e-6,
        period=20,
        n_max_gen=1000,
    )

    output = minimize(pymoo_problem,
                      algorithm,
                      termination,
                      verbose=True,
                      save_history=False)

    return output.X, output.F


class RustEval:
    variable_names = 'ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'

    def __init__(self, data, verbose=False):
        self.data = data
        d = data.T
        columns = []
        names = []
        for i in range(d.shape[0]-1):
            columns.append([float(v) for v in d[i]])
            names.append(RustEval.variable_names[i])
        target = [float(v) for v in d[-1]]
        self.verbose = verbose
        self.evaluator = Evaluator(columns, names, target)

    def evaluate(self, expression, constants=None):
        if constants is None:
            constants = []
        try:
            return self.evaluator.eval_expr(expression, constants)
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def get_error(self, expression, constants=None):
        if constants is None:
            constants = []
        try:
            return self.evaluator.get_rmse(expression, constants)
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def fit_and_evaluate(self, expr):
        num_constants = sum([1 for t in expr if t == "C"])
        if num_constants > 0:
            x, rmse = DE_pymoo(expr, num_constants, self.evaluator)
            return rmse, x
        else:
            return self.get_error(expr), []



if __name__ == '__main__':
    data = read_eq_data("/home/sebastianmeznar/Projects/HVAE/data/nguyen/nguyen10_test.csv")
    data = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]).T
    rev = RustEval(data)
    print(rev.fit_and_evaluate(["A", "C", "-"]))
# names = ["X", "Y"]
# evaluator = Evaluator(data, names)
# print(evaluator.eval_expr(["X", "Y", "+"]))