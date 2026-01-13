import torch
import math
import gpytorch
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.models.transforms.input import InputPerturbation
from botorch.acquisition.multi_objective.multi_output_risk_measures import MVaR
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval
from botorch.generation import MaxPosteriorSampling
from torch.quasirandom import SobolEngine
from multiprocessing import Pool, cpu_count
from Allocate_EMX import Rect_Opt_Flow

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
def perturb(X, perturbation_set):
    perturbation = InputPerturbation(perturbation_set=perturbation_set)
    return perturbation(X)

def make_single_point_hv_objective(ref_point):
    def hv_objective(Y, X=None):
        num_samples, n_candidates, m = Y.shape
        hv_values = torch.zeros(num_samples, n_candidates, device=Y.device)

        for i in range(num_samples):
            for j in range(n_candidates):
                front = Y[i, j:j+1, :]
                partitioning = DominatedPartitioning(ref_point=ref_point, Y=front)
                hv = float(partitioning.compute_hypervolume())
                hv_values[i, j] = hv
                # print(hv_values[i, j])
        return hv_values
    return hv_objective
def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y

def eval_problem_parallel(X: torch.Tensor, n_jobs=3) -> torch.Tensor:
    X_np = X.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]

    n_jobs = n_jobs or cpu_count()
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_simulate_single, x_list_set)

    return torch.tensor(results, dtype=torch.float64, device=X.device)

def evaluate_mvar_Y(X, eval_func, n_w, alpha, perturbation_set):
    perturbed_X = perturb(X, perturbation_set)
    perturbed_Y = eval_func(perturbed_X)
    mvar = MVaR(n_w=n_w, alpha=alpha)
    return mvar(perturbed_Y).view(-1, perturbed_Y.shape[-1])

def select_top_k_by_obj(Y_batch, k):
    scores = Y_batch.sum(dim=-1)
    return torch.topk(-scores, k=k).indices

def select_top_k_by_hv(Y_batch, current_frontier, ref_point, k):
    contribs = []
    for i in range(Y_batch.shape[0]):
        candidate_set = torch.cat([current_frontier, Y_batch[i:i+1]], dim=0)
        partitioning = DominatedPartitioning(ref_point=ref_point.to(**tkwargs), Y=candidate_set)
        hv = float(partitioning.compute_hypervolume())
        contribs.append(hv)
        # hv = DominatedPartitioning(ref_point=ref_point).compute_hypervolume(candidate_set)
        contribs.append(hv)
    contribs = torch.tensor(contribs)
    top_idx = torch.topk(contribs, k=k).indices
    return top_idx

def generate_batch_turbo(state, model, X, Y, batch_size, n_candidates=4096):
    ref_point = [-15, -15, -15]
    ref_point = torch.tensor(
        ref_point, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    print(x_center)
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    print(weights)
    tr_lb = torch.clamp(x_center - weights * state['length'] / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state['length'] / 2.0, 0.0, 1.0)
    print(tr_ub)
    print(tr_lb)


    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(X.device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, device=X.device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    hv_objective = make_single_point_hv_objective(ref_point=ref_point)
    thompson_sampling = MaxPosteriorSampling(
        model=model,
        objective=hv_objective,
        replacement=False
    )
    with torch.no_grad():
        posterior = model.posterior(X_cand)
        samples = posterior.rsample(torch.Size([batch_size]))
        # print(f"posterior samples shape: {samples.shape}")
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
        # print(X_next)
    return X_next

def update_turbo_state_by_hv(state, hv_prev, hv_now, delta=1e-4):
    improvement = hv_now > hv_prev + delta
    if improvement:
        state['success_counter'] += 1
        state['failure_counter'] = 0
    else:
        state['failure_counter'] += 1
        state['success_counter'] = 0

    if state['success_counter'] >= state['success_tolerance']:
        state['length'] = min(2.0 * state['length'], state['length_max'])
        state['success_counter'] = 0
    elif state['failure_counter'] >= state['failure_tolerance']:
        state['length'] /= 2.0
        state['failure_counter'] = 0

    if state['length'] < state['length_min']:
        state['restart_triggered'] = True

    return state

def turbo_mvar_solver(
    evaluate_func,
    bounds,
    N_INIT=128,
    BATCH_SIZE=50,
    SELECT_TOP_K=10,
    ALPHA=0.9,
    N_W=32,
    HV_REF=None,
    MAX_ITER=30,
):
    dim = len(bounds[-1])
    bounds_tensor = torch.tensor(bounds, **tkwargs)
    scaled_std = 0.1 / (bounds_tensor[1] - bounds_tensor[0])

    df = pd.read_csv("resultpsopara1.csv", header=None)
    df = df.iloc[:50]
    X_np = df.iloc[:, 0:13].to_numpy()
    Y_np = df[[18, 16, 14]].to_numpy()  # FoM22, FoM23, FoM24
    X = torch.tensor(X_np, **tkwargs)
    X = (X - bounds_tensor[0]) / (bounds_tensor[1] - bounds_tensor[0])
    Y = torch.tensor(Y_np, **tkwargs)
    # X = torch.rand(N_INIT, dim, **tkwargs)
    # X_scaled = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * X
    # Y = evaluate_func(X_scaled)

    pareto_Y = Y.clone()
    pareto_X = X.clone()
    state = {
        "length": 0.6,
        "length_min": 0.5**7,
        "length_max": 1.6,
        "success_counter": 0,
        "failure_counter": 0,
        "success_tolerance": 3,
        "failure_tolerance": 2,
        "restart_triggered": False,
        "best_value": Y.max().item(),
    }

    hv_history = []
    pareto_front = pareto_Y[is_non_dominated(pareto_Y)]
    partitioning = DominatedPartitioning(ref_point=HV_REF.to(**tkwargs), Y=pareto_front)
    hv_prev = float(partitioning.compute_hypervolume())
    hv_history.append(hv_prev)

    for iteration in range(MAX_ITER):
        perturbation_set = (
            torch.randn(N_W, dim, **tkwargs) * scaled_std
        )
        train_Y = (Y - Y.mean()) / Y.std()

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim))
        model = SingleTaskGP(X, train_Y, covar_module=covar_module, likelihood=likelihood)
        fit_gpytorch_mll(gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model))

        if iteration < 30:
            X_batch = generate_batch_turbo(state, model, X, train_Y, batch_size=BATCH_SIZE)
            X_batch_scaled = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * X_batch
            Y_batch = evaluate_func(X_batch_scaled)
            top_idx = select_top_k_by_obj(Y_batch, SELECT_TOP_K)
            selected_X = X_batch[top_idx]
            selected_Y = Y_batch[top_idx]
        else:
            X_batch = generate_batch_turbo(state, model, X, train_Y, batch_size=5)
            X_batch_scaled = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * X_batch
            mvar_Y_batch = evaluate_mvar_Y(X_batch_scaled, evaluate_func, N_W, ALPHA, perturbation_set)
            current_frontier = pareto_Y[is_non_dominated(pareto_Y)] if pareto_Y.numel() else mvar_Y_batch
            top_idx = select_top_k_by_hv(mvar_Y_batch, current_frontier, ref_point=HV_REF.to(**tkwargs), k=5)
            selected_X = X_batch[top_idx]
            selected_Y = mvar_Y_batch[top_idx]

        X = torch.cat([X, selected_X], dim=0)
        Y = torch.cat([Y, selected_Y], dim=0)

        pareto_Y = torch.cat([pareto_Y, selected_Y], dim=0)
        pareto_X = torch.cat([pareto_X, selected_X], dim=0)

        pareto_front = pareto_Y[is_non_dominated(pareto_Y)]
        # hv_now = DominatedPartitioning(ref_point=HV_REF.to(**tkwargs)).compute_hypervolume(pareto_front)
        partitioning = DominatedPartitioning(ref_point=HV_REF.to(**tkwargs), Y=pareto_front)
        hv_now = float(partitioning.compute_hypervolume())
        hv_history.append(hv_now)

        state = update_turbo_state_by_hv(state, hv_prev, hv_now)
        hv_prev = hv_now

        print(f"Iter {iteration+1}/{MAX_ITER}: Pareto size = {pareto_Y.shape[0]}, TR length = {state['length']:.4f}, HV = {hv_now:.4f}")

        if state['restart_triggered']:
            print("Trust region collapsed. Stopping.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    plt.figure(figsize=(10, 6))
    plt.plot(hv_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("MVaR Hypervolume Improvement Over Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pareto_X, pareto_Y

if __name__ == "__main__":
    bound_dict = {
        "FWn": [10, 300],
        "Fwp": [10, 300],
        "Fn_p": [1, 50],
        "Lp": [3, 20],
        "Mul_p": [1, 20],
        "Fn_n": [1, 50],
        "Ln": [3, 20],
        "Mul_n": [1, 20],
        "Cp": [1, 10000],
        "Cs": [1, 10000],
        "indp": [1, 6],
        "inds": [1, 6],
        "K": [0.1, 0.85]
    }
    bound = np.array(
        [
            [
                bound_dict["FWn"][0], bound_dict["Fwp"][0], bound_dict["Fn_p"][0], bound_dict["Lp"][0],
                bound_dict["Mul_p"][0], bound_dict["Fn_n"][0],
                bound_dict["Ln"][0], bound_dict["Mul_n"][0], bound_dict["Cp"][0], bound_dict["Cs"][0],
                bound_dict["indp"][0], bound_dict["inds"][0],
                bound_dict["K"][0]
            ]
            ,
            [
                bound_dict["FWn"][1], bound_dict["Fwp"][1], bound_dict["Fn_p"][1], bound_dict["Lp"][1],
                bound_dict["Mul_p"][1],
                bound_dict["Fn_n"][1], bound_dict["Ln"][1], bound_dict["Mul_n"][1], bound_dict["Cp"][1],
                bound_dict["Cs"][1],
                bound_dict["indp"][1], bound_dict["inds"][1], bound_dict["K"][1]
            ]
        ]
    )

    hv_ref = torch.tensor([-15.0, -15.0, -15.0])
    bounds = torch.tensor(bound, **tkwargs)
    pareto_X, pareto_Y = turbo_mvar_solver(
        evaluate_func=eval_problem_parallel,
        bounds=bounds,
        N_INIT=60,
        BATCH_SIZE=60,
        SELECT_TOP_K=10,
        ALPHA=0.9,
        N_W=2,
        HV_REF=hv_ref,
        MAX_ITER=30
    )

    print("Final Pareto Points (designs):", pareto_X)
    print("Final Pareto Objectives:", pareto_Y)
