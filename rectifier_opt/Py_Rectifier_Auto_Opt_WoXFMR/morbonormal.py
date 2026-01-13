import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform, ScalarizedObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.sampling import draw_sobol_samples
from torch.multiprocessing import Pool, cpu_count
from Allocate_EMX import Rect_Opt_Flow
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
tkwargs = {"dtype": dtype, "device": device}
input_dim, output_dim = 13, 32
dim=13

bound_dict={
    "FWn":              [10, 300],
    "Fwp":              [10, 300],
    "Fn_p":             [1, 50],
    "Lp":               [3, 20],
    "Mul_p":            [1, 20],
    "Fn_n":             [1, 50],
    "Ln":               [3, 20],
    "Mul_n":            [1, 20],
    "Cp":               [1, 10000],
    "Cs":               [1, 10000],
    "indp":             [1, 6],
    "inds":             [1, 6],
    "K":                [0.1, 0.85]
}
bound = np.array([[v[0] for v in bound_dict.values()],
                  [v[1] for v in bound_dict.values()]])
bounds = torch.tensor(bound, **tkwargs)
bound_list = list(bound_dict.values())
bounds_real = torch.tensor(bound_list, dtype=torch.float64).T
def _simulate_single(x_list):
    y = Rect_Opt_Flow(x_list)
    return y

def eval_problem_parallel(X: torch.Tensor, n_jobs=40) -> torch.Tensor:
    X_np = X.cpu().numpy()
    x_list_set = [x.tolist() for x in X_np]
    with Pool(processes=n_jobs or cpu_count()) as pool:
        results = pool.map(_simulate_single, x_list_set)
    return torch.tensor(results, dtype=dtype, device=X.device)

n_init = 40
# X_normalized = draw_sobol_samples(bounds=torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float64),
#                                    n=1, q=n_init).squeeze(0)
# X = unnormalize(X_normalized, bounds_real)
# print(X)
# Y = eval_problem_parallel(X)

df = pd.read_csv("/home/ieda/wangl/PY/Py_Rectifier_Auto_Opt_WoXFMR/result/mobo 32p.csv")
df = df.iloc[:40]
X = torch.tensor(df.iloc[:, :13].values, dtype=torch.float64)
Y = torch.tensor(df.iloc[:, -32:].values, dtype=torch.float64)
X = X.to(**tkwargs)
Y = Y.to(**tkwargs)
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
q = 40
num_iterations = 30
beta = 0.2

for iter in range(num_iterations):
    print(f"\n[Iter {iter}] Num data: {X.shape[0]}")

    models = []
    for i in range(output_dim):
        model = SingleTaskGP(
            X, Y[:, i:i+1],
            input_transform=Normalize(d=input_dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        models.append(model)
    model_list = ModelListGP(*models)

    weights = torch.rand(output_dim, **tkwargs)
    posterior_transform = ScalarizedPosteriorTransform(weights=weights)
    # objective = ScalarizedObjective(weights=weights)
    ucb = qUpperConfidenceBound(model=model_list, beta=0.1, sampler=sampler, posterior_transform=posterior_transform)
    candidates, _ = optimize_acqf(
        acq_function=ucb,
        bounds=bounds,
        q=q,
        num_restarts=20,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 100},
    )
    candidate = candidates.to(X.device)
    print("sele")
    # print(candidate)
    Y_new = eval_problem_parallel(candidate)
    Y_new = Y_new.to(Y.device)
    X = torch.cat([X, candidate], dim=0)
    Y = torch.cat([Y, Y_new], dim=0)
