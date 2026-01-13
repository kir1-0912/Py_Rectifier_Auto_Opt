import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from trust_region import ScalarizedTrustRegion, TurboHParams
from Py_Rectifier_Auto_Opt_WoXFMR.Allocate_EMX import Rect_Opt_Flow
import pandas as pd
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler


d = 13
dim = 13
m = 32
n_init = 40
batch_size = 40
init_points = 100
max_iter = 30
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

bound_dict = {
    "FWn": [10, 300], "Fwp": [10, 300], "Fn_p": [1, 50], "Lp": [3, 20],
    "Mul_p": [1, 20], "Fn_n": [1, 50], "Ln": [3, 20], "Mul_n": [1, 20],
    "Cp": [1, 10000], "Cs": [1, 10000], "indp": [1, 6], "inds": [1, 6],
    "K": [0.1, 0.85]
}
bound_list = list(bound_dict.values())
bounds_real = torch.tensor(bound_list, dtype=torch.float64).T  # shape: (2, d)
df = pd.read_csv("/home/ieda/wangl/PY/Py_Rectifier_Auto_Opt_WoXFMR/result/mobo 32p.csv")
df = df.iloc[:40]
X_init_real = torch.tensor(df.iloc[:, :13].values, dtype=torch.float64)
Y_init = torch.tensor(df.iloc[:, -32:].values, dtype=torch.float64)
X_init = normalize(X_init_real, bounds_real)
bounds_unit = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float64)
# weights = sample_simplex(d=m, n=1, dtype=Y_init.dtype, device=Y_init.device).squeeze(0)
weights = torch.ones(m, dtype=Y_init.dtype, device=Y_init.device)
# important_dims = [0, 7, 8, 15, 16, 23, 24, 31]
# weights[important_dims] = 5.0
weights = weights / weights.sum()


def generate_batch_ucb(
        model,
        bounds: torch.Tensor,
        q: int = 40,
        beta: float = 0.2,
        weights =weights
) -> torch.Tensor:
    if weights is None:
        raise ValueError("Must provide weights for scalarizing multi-output model.")

    posterior_transform = ScalarizedPosteriorTransform(weights=weights)

    ucb = qUpperConfidenceBound(model=model, beta=0.1, sampler=sampler, posterior_transform=posterior_transform)

    X_batch, _ = optimize_acqf(
        acq_function=ucb,
        bounds=bounds,
        q=q,
        num_restarts=10,
        raw_samples=128,
        return_best_only=True,
    )
    return X_batch

def _simulate_single(x_list):
    return Rect_Opt_Flow(x_list)

def eval_problem_parallel(X_unit: torch.Tensor,
                          n_jobs=40) -> torch.Tensor:
    X_real = unnormalize(X_unit, bounds_real)
    from multiprocessing import Pool, cpu_count
    X_np = X_real.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]
    with Pool(processes=n_jobs or cpu_count()) as pool:
        results = pool.map(_simulate_single, x_list_set)
    return torch.tensor(results, dtype=torch.float64, device=X_unit.device)

def multi_obj_fn(X_unit: torch.Tensor) -> torch.Tensor:
    return eval_problem_parallel(X_unit)[:, :32]

# df = pd.read_csv("/home/ieda/lvhq/Py_Rectifier_Auto_Opt_WoXFMR/morbo/morbo/resultinitw.csv")
# df = df.iloc[:40]
# X_init_real = torch.tensor(df.iloc[:, :13].values, dtype=torch.float64)
# Y_init = torch.tensor(df.iloc[:, -32:].values, dtype=torch.float64)
# X_normalized = draw_sobol_samples(bounds=torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float64),
#                                    n=1, q=n_init).squeeze(0)
# X_init_real = unnormalize(X_normalized, bounds_real)
# X_init = normalize(X_init_real, bounds_real)
# Y_init = eval_problem_parallel(X_init)
# print("X_init_real min:", X_init_real.min(dim=0).values)
# print("X_init_real max:", X_init_real.max(dim=0).values)
# print("bounds_real:", bounds_real)


def objective(Y: torch.Tensor) -> torch.Tensor:
    return Y



tr_hparams = TurboHParams(
    success_streak=3,
    failure_streak=3,
    eps=0.001,
    length_init=0.3,
    length_min=0.5**10,
    length_max=1.6,
    use_ard=True,
    track_history=True,
    verbose=True,
)

tr_solver = ScalarizedTrustRegion(
    X_init=X_init,
    Y_init=Y_init,
    bounds=bounds_unit,
    tr_hparams=tr_hparams,
    objective=objective,
    weights=weights,
)


for iteration in range(max_iter):
    print(f"\n[Iter {iteration}] Generating batch...")
    tr_bounds = tr_solver.get_bounds(model_space=True)
    print(tr_bounds)
    if tr_solver.model is None:
        X_cand_unit = draw_sobol_samples(bounds=tr_bounds, n=1, q=batch_size).squeeze(0)
        print(1)
    else:
        X_cand_unit = generate_batch_ucb(
            model=tr_solver.model,
            bounds=tr_bounds,
            q=batch_size,
            beta=0.2,
        )
        print(2)
    X_cand_real = unnormalize(X_cand_unit, bounds_real)
    X_cand_unit = normalize(X_cand_real, bounds_real)
    print(f"[Iter {iteration}] Evaluating candidates...")
    Y_cand = eval_problem_parallel(X_cand_unit)
    print(f"[Iter {iteration}] Evaluation done ")
    X_all = torch.cat([tr_solver.X, X_cand_unit], dim=0)
    Y_all = torch.cat([tr_solver.Y, Y_cand], dim=0)
    restart = tr_solver.update(
        X_all=X_all,
        Y_all=Y_all,
        X_new=X_cand_unit,
        Y_new=Y_cand,
    )
    if restart:
        print(f"[Iter {iteration}] Restart triggered due to trust region too small.")
        break

best_idx = tr_solver.objective(tr_solver.Y_estimate).argmax().item()
best_x = unnormalize(tr_solver.X[best_idx:best_idx + 1], bounds_real)
best_y = tr_solver.Y[best_idx:best_idx + 1]

print("Best input (real scale):", best_x)
print("Best objective(s):", best_y)