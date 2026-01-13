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
from design import Rect_Opt_Flow
from multiprocessing import Pool, cpu_count


tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

bound_dict = {
    "C1": [0.1, 5],
    "C2": [0.1, 5],
    "L1": [0.1, 5],
    "K": [0.2, 0.8],
    "L2": [1, 10],
    "N": [1, 200]
}

bound = np.array([
    [bound_dict[key][0] for key in bound_dict],
    [bound_dict[key][1] for key in bound_dict]
])

# def eval_problem(X_np: np.ndarray) -> torch.Tensor:
#     results = []
#     for x in X_np:
#         x_list = x.tolist()
#         y = Rect_Opt_Flow(x_list)
#         results.append(y)
#     return torch.tensor(results, dtype=torch.float64)
def enforce_custom_constraint(X):
    eps = 1e-8
    X[:, 1] = 2.0 * X[:, 0] * X[:, 3] / (X[:, 4] + eps)
    return X

def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y

def find_max_hv_contributor(Y, ref_point):
    mask = is_non_dominated(Y)
    nd_Y = Y[mask]

    partition = DominatedPartitioning(ref_point=ref_point)
    partition.update(nd_Y)
    full_hv = partition.compute_hypervolume()

    max_contrib = -float("inf")
    best_idx = -1
    for i in range(nd_Y.shape[0]):
        subset = torch.cat([nd_Y[:i], nd_Y[i + 1:]], dim=0)
        sub_partition = DominatedPartitioning(ref_point=ref_point)
        sub_partition.update(subset)
        hv_without_i = sub_partition.compute_hypervolume()
        contrib = full_hv - hv_without_i
        if contrib > max_contrib:
            max_contrib = contrib
            best_idx = i
    print(max_contrib)
    true_indices = torch.arange(Y.shape[0], device=Y.device)[mask]
    return nd_Y[best_idx], true_indices[best_idx].item()
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
        # print("All crowding distances are inf. Using first objective.")
        best_idx = torch.argmax(nd_Y[:, 0]).item()
    else:
        best_idx = torch.argmax(cd).item()

    return nd_X[best_idx], cd, mask


def FHE_MOPSO_CD_step(X, V, P_best, P_best_Y, eval_fn, bounds):
    Y = eval_fn(X)
    # P_all = torch.cat([P_best, X], dim=0)
    # P_all_Y = torch.cat([P_best_Y, Y], dim=0)
    # mask = is_non_dominated(P_all_Y)
    # P_best = P_all[mask]
    # P_best_Y = P_all_Y[mask]
    for i in range(X.shape[0]):
        if dominates(Y[i], P_best_Y[i]):
            P_best[i] = X[i]
            P_best_Y[i] = Y[i]
    G_best, cd, _ = select_global_best(P_best_Y, P_best)
    # print(G_best)
    # print(cd)
    # print(G_best.shape)
    X, V = pso_update(X, V, P_best, G_best.unsqueeze(0).expand_as(X))
    X = torch.where(X < bounds[0], bounds[0] + 0.01 * (bounds[1] - bounds[0]), X)
    X = torch.where(X > bounds[1], bounds[1] - 0.01 * (bounds[1] - bounds[0]), X)
    # X = enforce_custom_constraint(X)
    return X, V, P_best, P_best_Y, G_best, cd, Y

def initialize_particles(n, bounds):
    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand((n, bounds.shape[1]), **tkwargs)
    V = torch.zeros_like(X)
    return X, V

# Settings
n_particles = 40
max_iters = 30
dim = 5
bounds = torch.tensor(bound, **tkwargs)
ref_point = torch.tensor([-1, -1, -1], **tkwargs)


# Initialize
# df = pd.read_csv("resultpsof2and3.csv", header=None)
# df = df.iloc[:60]
# X_np = df.iloc[:, 0:13].to_numpy()
# Y_np = df[[16, 14]].to_numpy()  # FoM22, FoM23, FoM24
# X = torch.tensor(X_np, **tkwargs)
# V = torch.zeros_like(X)
X, V = initialize_particles(n_particles, bounds)
# print(X)
# X = enforce_custom_constraint(X)
P_best = X.clone()
Y_np = eval_problem_parallel(P_best)
P_best_Y = torch.tensor(Y_np, **tkwargs)
P_best_Y = P_best_Y.to(ref_point.device)
best_y, idx = find_max_hv_contributor(P_best_Y, ref_point)
print(best_y)
print(P_best[idx])
all_hv = []
mvar_hv = DominatedPartitioning(ref_point=ref_point)
mvar_hv.update(P_best_Y)
hv = mvar_hv.compute_hypervolume().item()
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
        X, V, P_best, P_best_Y, eval_problem_parallel, bounds
    )
    if t % itn == 0:
        worst_idx = torch.argmin(cd)
        dim_to_mutate = torch.randint(X.shape[1], (1,))
        X[worst_idx, dim_to_mutate] = bounds[0, dim_to_mutate] + (
                bounds[1, dim_to_mutate] - bounds[0, dim_to_mutate]
        ) * torch.rand(1, **tkwargs)
        X[worst_idx] = torch.min(torch.max(X[worst_idx], bounds[0]), bounds[1])
        # X[worst_idx] = enforce_custom_constraint(X[worst_idx].unsqueeze(0)).squeeze(0)
    P_best_Y = P_best_Y.to(ref_point.device)
    best_y, idx = find_max_hv_contributor(P_best_Y, ref_point)
    print(best_y)
    print(P_best[idx])
    mvar_hv = DominatedPartitioning(ref_point=ref_point)
    mvar_hv.update(P_best_Y)
    hv = mvar_hv.compute_hypervolume().item()
    all_hv.append(hv)
    print(f"Iter {t}: HV = {hv:.4f}")

# Plot HV progression
plt.figure()
plt.plot(all_hv)
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.title("MVaR Hypervolume over Iterations")
plt.grid(True)
plt.show()