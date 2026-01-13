import numpy as np
import torch
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from allo.ALL import Rect_Opt_Flow
from multiprocessing import Pool, cpu_count



def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y

def eval_problem_parallel(X: torch.Tensor, n_jobs=30) -> torch.Tensor:
    X_np = X if isinstance(X, np.ndarray) else X.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]
    n_jobs = n_jobs or cpu_count()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_simulate_single, x_list_set)
    return torch.tensor(results)


class MYProblem(Problem):
    def __init__(self):
        super().__init__(n_var=13,
                         n_obj=24,
                         n_constr=0,
                         xl=[10, 10, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 0.1],
                         xu=[300, 300, 50, 20, 20, 50, 20, 20, 10000, 10000, 6, 6, 0.85],
                         elementwise_evaluation=False)

    def _evaluate(self, X, out, *args, **kwargs):
        y = -eval_problem_parallel(X)
        out["F"] = np.array(y)



class MyCallback(Callback):
    def __init__(self):
        super().__init__()

    def notify(self, algorithm):
        print(f"Generation {algorithm.n_gen} completed.")


def random_ref_dirs(n_obj=24, n_points=30):
    dirs = np.random.rand(n_points, n_obj)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs

if __name__ == "__main__":
    ref_dirs = random_ref_dirs(n_obj=24, n_points=40)

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )

    problem = MYProblem()
    callback = MyCallback()

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=('n_gen', 30),
                   callback=callback,
                   seed=1,
                   verbose=False,
                   save_history=True)

    plot = Scatter()
    plot.add(res.F[:, :2], color="red")
    plot.show()


