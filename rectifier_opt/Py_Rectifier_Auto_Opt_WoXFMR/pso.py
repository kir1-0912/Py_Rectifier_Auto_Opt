import torch
import numpy as np
import pandas as pd
import os
from botorch.test_functions.multi_objective import ToyRobust
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.multi_objective.multi_output_risk_measures import MVaR
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.monte_carlo import qSimpleRegret
from Allocate_EMX import Rect_Opt_Flow
from multiprocessing import Pool, cpu_count


tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

bound_dict={
    "FWn":              [10,300],
    "Fwp":              [10,300],
    "Fn_p":             [1,50],
    "Lp":               [3,20],
    "Mul_p":            [1,20],
    "Fn_n":             [1,50],
    "Ln":               [3,20],
    "Mul_n":            [1,20],
    "Cp":               [1,10000],
    "Cs":               [1,10000],
    "indp":             [1,6],
    "inds":             [1,6],
    "K":                [0.1,0.85]
}
bound = np.array(
                [
                      [
                        bound_dict["FWn"][0],bound_dict["Fwp"][0],bound_dict["Fn_p"][0],bound_dict["Lp"][0], bound_dict["Mul_p"][0], bound_dict["Fn_n"][0],
                        bound_dict["Ln"][0],bound_dict["Mul_n"][0],bound_dict["Cp"][0],bound_dict["Cs"][0], bound_dict["indp"][0],bound_dict["inds"][0],
                        bound_dict["K"][0]
                      ]
                  ,
                    [
                        bound_dict["FWn"][1], bound_dict["Fwp"][1], bound_dict["Fn_p"][1], bound_dict["Lp"][1], bound_dict["Mul_p"][1],
                        bound_dict["Fn_n"][1],bound_dict["Ln"][1], bound_dict["Mul_n"][1], bound_dict["Cp"][1], bound_dict["Cs"][1],
                        bound_dict["indp"][1],bound_dict["inds"][1],bound_dict["K"][1]
                    ]
                  ]
                 )

# def eval_problem(X_np: np.ndarray) -> torch.Tensor:
#     results = []
#     for x in X_np:
#         x_list = x.tolist()
#         y = Rect_Opt_Flow(x_list)
#         results.append(y)
#     return torch.tensor(results, dtype=torch.float64)

def compute_grouped_hv(Y, group_sizes, subset_size=5):

    n_obj = Y.shape[1]
    idx = 0
    total_hv = 0.0

    for group_size in group_sizes:
        group_Y = Y[:, idx:idx + group_size]
        idx += group_size

        for _ in range(1):
            selected_idx = torch.randperm(group_size)[:subset_size]
            sub_Y = group_Y[:, selected_idx]
            ref_point = torch.full((subset_size,), -5.0, **tkwargs)

            try:
                part = DominatedPartitioning(ref_point=ref_point)
                part.update(sub_Y)
                hv = part.compute_hypervolume().item()
                total_hv += hv
            except:
                total_hv += 0.0

    return total_hv

def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y

def find_max_hv_contributor(Y, group_sizes, subset_size=5):
    mask = is_non_dominated(Y)
    nd_Y = Y[mask]

    idx_map = torch.arange(Y.shape[0], device=Y.device)[mask]
    max_contrib = -float("inf")
    best_idx = -1
    for i in range(nd_Y.shape[0]):
        subset = torch.cat([nd_Y[:i], nd_Y[i + 1:]], dim=0)
        hv = compute_grouped_hv(subset, group_sizes, subset_size)
        contrib = compute_grouped_hv(nd_Y, group_sizes, subset_size) - hv
        if contrib > max_contrib:
            max_contrib = contrib
            best_idx = i

    return nd_Y[best_idx], idx_map[best_idx].item()

def dominates(y1, y2):
    return torch.all(y1 >= y2) and torch.any(y1 > y2)

def eval_problem_parallel(X: torch.Tensor, n_jobs=40) -> torch.Tensor:
    X_np = X.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]

    n_jobs = n_jobs or cpu_count()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_simulate_single, x_list_set)

    return torch.tensor(results, dtype=torch.float64, device=X.device)



def crowding_distance(Y):
    n, m = Y.shape
    dist = torch.zeros(n, dtype=Y.dtype, device=Y.device)
    for i in range(m):
        sorted_idx = torch.argsort(Y[:, i])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        norm = Y[sorted_idx[-1], i] - Y[sorted_idx[0], i]
        if norm == 0:
            continue
        for j in range(1, n - 1):
            dist[sorted_idx[j]] += (
                    (Y[sorted_idx[j + 1], i] - Y[sorted_idx[j - 1], i]) / norm
            )
    return dist


def pso_update(X, V, P_best, G_best, w=0.4, c1=1.5, c2=1.5):
    r1 = torch.rand_like(X)
    r2 = torch.rand_like(X)
    # print("X.shape:", X.shape)
    # print("P_best.shape:", P_best.shape)
    # print("G_best.shape:", G_best.shape)
    # print("r1.shape:", r1.shape)
    # print("r2.shape:", r2.shape)
    V_new = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
    X_new = X + V_new
    return X_new, V_new


def select_global_best(Y, X):
    # print('yshape', Y.shape)
    mask = is_non_dominated(Y)
    nd_X = X[mask]
    nd_Y = Y[mask]
    # print('ndyshape', nd_Y.shape)
    cd = crowding_distance(nd_Y)
    if torch.all(torch.isinf(cd)):
        best_idx = torch.argmax(nd_Y[:, 0]).item()
    else:
        best_idx = torch.argmax(cd).item()

    return nd_X[best_idx], cd, mask

def select_global_best_adaptive(P_best_Y, P_best, group_sizes, subset_size, iter_idx, switch_iter=20):
    if iter_idx < switch_iter:
        mask = is_non_dominated(P_best_Y)
        nd_X = P_best[mask]
        nd_Y = P_best_Y[mask]
        cd = crowding_distance(nd_Y)
        if torch.all(torch.isinf(cd)):
            best_idx = torch.argmax(nd_Y[:, 0]).item()
        else:
            best_idx = torch.argmax(cd).item()
        return nd_X[best_idx], cd, mask
    else:
        best_y, idx = find_max_hv_contributor(P_best_Y, group_sizes, subset_size)
        cd = crowding_distance(P_best_Y)
        return P_best[idx], cd, is_non_dominated(P_best_Y)

def FHE_MOPSO_CD_step(X, V, P_best, P_best_Y, eval_fn, bounds, t, group_sizes, subset_size=4):
    Y = eval_fn(X)

    for i in range(X.shape[0]):
        current_contrib = compute_grouped_hv(P_best_Y[i].unsqueeze(0), group_sizes, subset_size)
        new_contrib = compute_grouped_hv(Y[i].unsqueeze(0), group_sizes, subset_size)
        if dominates(Y[i], P_best_Y[i]) or new_contrib > current_contrib:
            P_best[i] = X[i]
            P_best_Y[i] = Y[i]

    G_best, cd, _ = select_global_best_adaptive(P_best_Y, P_best, group_sizes, subset_size, t)

    X, V = pso_update(X, V, P_best, G_best.unsqueeze(0).expand_as(X))
    X = torch.where(X < bounds[0], bounds[0] + 0.01 * (bounds[1] - bounds[0]), X)
    X = torch.where(X > bounds[1], bounds[1] - 0.01 * (bounds[1] - bounds[0]), X)
    return X, V, P_best, P_best_Y, G_best, cd, Y

def initialize_particles(n, bounds):
    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand((n, bounds.shape[1]), **tkwargs)
    V = torch.zeros_like(X)
    return X, V

# Settings
n_particles = 40
max_iters = 30
#dim = 30
bounds = torch.tensor(bound, **tkwargs)
# ref_point = torch.full((24,), -2.0, **tkwargs)


# Initialize
# df = pd.read_csv("resultpsof2and3.csv", header=None)
# df = df.iloc[:60]
# X_np = df.iloc[:, 0:13].to_numpy()
# Y_np = df[[16, 14]].to_numpy()  # FoM22, FoM23, FoM24
# X = torch.tensor(X_np, **tkwargs)
# V = torch.zeros_like(X)
X, V = initialize_particles(n_particles, bounds)
P_best = X.clone()
Y_np = eval_problem_parallel(P_best)
P_best_Y = torch.tensor(Y_np, **tkwargs)
group_sizes = [8, 8, 8, 8]
# P_best_Y = P_best_Y.to(group_sizes.device)
best_y, idx = find_max_hv_contributor(P_best_Y, group_sizes=group_sizes, subset_size=4)
print(best_y)
print(P_best[idx])
all_hv = []
hv = compute_grouped_hv(P_best_Y, group_sizes=group_sizes, subset_size=5)
print(hv)
all_hv.append(hv)
G_best, cd, _ = select_global_best(P_best_Y, P_best)
print(G_best)
print(cd)
X, V = pso_update(X, V, P_best, G_best.unsqueeze(0).expand_as(X))
X = torch.where(X < bounds[0], bounds[0] + 0.01 * (bounds[1] - bounds[0]), X)
X = torch.where(X > bounds[1], bounds[1] - 0.01 * (bounds[1] - bounds[0]), X)

# df = pd.read_csv("resultpsopara.csv", header=None)
# df = df.iloc[:50]
# X_np = df.iloc[:, 0:13].to_numpy()
# Y_np = df[[18, 16, 14]].to_numpy()  # FoM22, FoM23, FoM24
# X = torch.tensor(X_np, **tkwargs)
# V = torch.zeros_like(X)
# P_best = X.clone()
# P_best_Y = torch.tensor(Y_np, **tkwargs)
# P_best_Y = P_best_Y.to(ref_point.device)
# mvar_hv = DominatedPartitioning(ref_point=ref_point)
# mvar_hv.update(P_best_Y)
# all_hv = [mvar_hv.compute_hypervolume().item()]

itn=5
# Optimization loop
for t in range(max_iters):
    X, V, P_best, P_best_Y, G_best, cd, Y = FHE_MOPSO_CD_step(
        X, V, P_best, P_best_Y, eval_problem_parallel, bounds, t, group_sizes, subset_size=4
    )

    if t % itn == 0:
        worst_idx = torch.argmin(cd)
        dim_to_mutate = torch.randint(X.shape[1], (1,))
        X[worst_idx, dim_to_mutate] = bounds[0, dim_to_mutate] + (
                bounds[1, dim_to_mutate] - bounds[0, dim_to_mutate]
        ) * torch.rand(1, **tkwargs)
        X[worst_idx] = torch.min(torch.max(X[worst_idx], bounds[0]), bounds[1])

    best_y, idx = find_max_hv_contributor(P_best_Y, group_sizes=group_sizes, subset_size=4)
    print(best_y)
    print(P_best[idx])
    hv = compute_grouped_hv(P_best_Y, group_sizes=group_sizes, subset_size=4)
    print(f"Iter {t}: Grouped HV = {hv:.4e}")
    all_hv.append(hv)

# Plot HV progression
plt.figure()
plt.plot(all_hv)
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.title("MVaR Hypervolume over Iterations")
plt.grid(True)
plt.show()




