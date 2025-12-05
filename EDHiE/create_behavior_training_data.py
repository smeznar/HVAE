from SRToolkit.dataset import SRBenchmark
from SRToolkit.utils import generate_n_expressions, create_behavior_matrix, bed
import numpy as np

from argparse import ArgumentParser
from copy import copy

if __name__ == '__main__':
    parser = ArgumentParser(prog='Symbolic regression', description='Run a symbolic regression benchmark')
    parser.add_argument("-i", type=int)
    args = parser.parse_args()
    # num_constants = 32
    # benchmark = SRBenchmark.feynman("../data/fey_data")
    # dataset_name = benchmark.list_datasets(num_variables=2, verbose=False)[0]
    # dataset = benchmark.create_dataset(dataset_name)
    # sl = dataset.symbols
    # x = dataset.X[:64]
    # expressions = generate_n_expressions(sl, 5000, max_expression_length=30, unique=True)
    # inds = [i for i in range(len(expressions)) if "C" in expressions[i]]
    # rs_exp = np.random.choice(inds, 5000, replace=True)
    # expressions = expressions + [copy(expressions[ind]) for ind in rs_exp]
    #
    # behavior_matrices = []
    # for expr in expressions:
    #     bm = create_behavior_matrix(expr, x, num_consts_sampled=num_constants, consts_bounds=(0.2, 5), symbol_library=sl)
    #     if bm.shape[1] == 1:
    #         bm = np.repeat(bm, num_constants, axis=1)
    #     behavior_matrices.append(bm)
    #
    # behavior_matrices = np.array(behavior_matrices)
    # np.save(f"../data/behavior_matrices_test.npy", behavior_matrices)
    # behavior_matrices = np.load(f"../data/behavior_matrices_test.npy")
    # distances = np.zeros((1, behavior_matrices.shape[0]))
    # for j in range(args.i+1, behavior_matrices.shape[0]):
    #     distances[0, j] = bed(behavior_matrices[args.i], behavior_matrices[j], X=np.zeros((64,32)))
    # np.save(f"../data/behavior_distances_test_{args.i}.npy", distances)
    distances = np.zeros((10000, 10000))
    for i in range(10000):
        distances[i, :] = np.load(f"../data/behavior_distances_test_{i}.npy")[0]

    a = 0
    for i in range(10000):
        for j in range(i+1, 10000):
            distances[j, i] = distances[i, j]

    np.save(f"../data/behavior_distances_test.npy", distances)