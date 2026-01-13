import numpy as np
import multiprocessing as mp
import torch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback



class CircuitOptimizationProblem(Problem):
    def __init__(self, evaluate_func, bounds):
        super().__init__(n_var=bounds.shape[1], n_obj=1, xl=bounds[0], xu=bounds[1])
        self.evaluate_func = evaluate_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluate_func(x)

class CustomOutput(Callback):
    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.data["n_evals"] = []
        self.data["best"] = []
        self.data["avg"] = []
        self.verbose = verbose

    def notify(self, algorithm):
        self.data["n_evals"].append(algorithm.evaluator.n_eval)
        self.data["best"].append(algorithm.opt.get("F")[0])
        self.data["avg"].append(np.mean(algorithm.pop.get("F")))

        if self.verbose:
            n_gen = algorithm.n_gen
            n_evals = algorithm.evaluator.n_eval
            best = self.data["best"][-1]
            avg = self.data["avg"][-1]

            # 确保 best 和 avg 是标量
            if isinstance(best, np.ndarray):
                best = best.item()
            if isinstance(avg, np.ndarray):
                avg = avg.item()

            print(f"{n_gen:>4d} | {n_evals:>7d} | {best:13.6f} | {avg:13.6f}")

def cmaes_solver(
    parallel_evaluate,
    bounds,
    x0=None,
    sigma=0.25,
    restarts=3,
    restart_from_best=True,
    popsize=20,
    max_generations=1000,
    verbose=True
):
    """
    CMAES求解器

    参数:
    parallel_evaluate: 并行评估函数
    bounds: 边界条件，形如 [[lb1, lb2, ...], [ub1, ub2, ...]]
    x0: 初始解，如果为None则使用边界的平均值
    sigma: CMA-ES的初始步长
    restarts: 重启次数
    restart_from_best: 是否从最佳解重启
    popsize: 种群大小
    max_generations: 最大迭代次数
    verbose: 是否打印详细信息

    返回:
    best_x: 最优解
    best_y: 最优解对应的函数值
    """
    def evaluate_wrapper(x):
        return parallel_evaluate(x).cpu().numpy()

    bounds = np.array(bounds)
    problem = CircuitOptimizationProblem(evaluate_wrapper, bounds)

    if x0 is None:
        x0 = np.mean(bounds, axis=0)

    algorithm = CMAES(
        x0=x0,
        sigma=sigma,
        restarts=restarts,
        restart_from_best=restart_from_best,
        popsize=popsize
    )

    if verbose:
        print("  Gen |   Evals |          Best |           Avg")
        print("-------------------------------------------------")

    res = minimize(problem,
                   algorithm,
                   ('n_gen', max_generations),
                   callback=CustomOutput(verbose=verbose),
                   verbose=False)

    best_x = res.X
    best_y = res.F[0]

    if verbose:
        print("\nFinal Results:")
        print("Best parameters:")
        for i, param in enumerate(best_x):
            print(f"Param {i + 1}: {param:.6f}")
        print(f"Best objective value: {best_y:.6f}")

    return best_x, best_y

