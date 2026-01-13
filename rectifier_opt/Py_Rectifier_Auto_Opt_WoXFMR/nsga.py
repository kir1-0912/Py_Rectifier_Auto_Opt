import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter
from design import Rect_Opt_Flow
from multiprocessing import Pool, cpu_count
# print(FloatRandomSampling)

def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y


def eval_problem_parallel(X: torch.Tensor, n_jobs=40) -> torch.Tensor:
    X_np = X if isinstance(X, np.ndarray) else X.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]

    n_jobs = n_jobs or cpu_count()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_simulate_single, x_list_set)

    return torch.tensor(results)

def compute_grouped_hv_sum(F, group_sizes=[8, 8, 8], subset_size=4, ref_val=10.0):
    assert F.shape[1] == sum(group_sizes), "Mismatch between total objectives and group_sizes."
    idx = 0
    total_hv = 0.0
    for group_size in group_sizes:
        group_F = F[:, idx:idx + group_size]
        idx += group_size
        selected_idx = np.random.choice(group_size, size=subset_size, replace=False)
        sub_F = group_F[:, selected_idx]
        try:
            hv = HV(ref_point=np.full(subset_size, ref_val)).do(sub_F)
            total_hv += hv
        except Exception as e:
            print(f"[Warning] HV computation failed: {e}")
            total_hv += 0.0
    return total_hv


class MYProblem(Problem):
    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=3,
                         n_constr=0,
                         xl=[0.1, 0.1, 0.1, 0.2, 1, 1],
                         xu=[5, 5, 5, 0.8, 10, 200],
                         elementwise_evaluation=False)

    def _evaluate(self, X, out, *args, **kwargs):
        y = -eval_problem_parallel(X)
        F = np.array(y)
        out["F"] = np.array(F)



# class MyCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.max_hv_history = []
#
#     def notify(self, algorithm):
#         F = algorithm.opt.get("F")
#         hv = compute_grouped_hv_sum(F, group_sizes=[8, 8, 8], subset_size=4, ref_val=3.0)
#         if len(self.max_hv_history) == 0 or hv > max(self.max_hv_history):
#             self.max_hv_history.append(hv)
#             print(f"Generation {algorithm.n_gen}: New Max Grouped HV = {hv:.6f}")
#         else:
#             self.max_hv_history.append(self.max_hv_history[-1])
#             print(f"Generation {algorithm.n_gen}: Current Max Grouped HV = {self.max_hv_history[-1]:.6f}")


algorithm = NSGA2(
    pop_size=40,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# callback = MyCallback()
problem = MYProblem()
res = minimize(problem=problem,
               algorithm=algorithm,
               termination=('n_gen', 30),
               seed=1,
               verbose=False,
               save_history=True)

# print("\nHV History:")
# for i, hv in enumerate(callback.max_hv_history):
#     print(f"Gen {i+1}: {hv:.6f}")

plot = Scatter()
plot.add(res.F, color="red")
plot.show()