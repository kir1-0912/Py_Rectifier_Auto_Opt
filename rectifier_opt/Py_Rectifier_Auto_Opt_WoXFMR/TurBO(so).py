import os
import math
import pickle
import warnings
from dataclasses import dataclass
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from gpytorch.kernels import Kernel
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils import *
from botorch.settings import debug
from torch.nn import Parameter

device: torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tkwargs = {
#     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     "dtype": torch.double,
# }
#
# DIM = len(bound_op[-1])  # 电路参数的维度
# N_INIT = 256  # 初始采样点数,之前是设成了2*维度
# N_ITERATIONS = 30  # 迭代次数
# BATCH_SIZE = 20  # 每次迭代的批量大小,如果用EI，就不能设置得很大
#
# bounds_tensor = torch.tensor(bound_op)



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8  # 信赖域的当前长度
    length_min: float = 0.5 ** 7  # 信赖域长度的最小和最大值。
    length_max: float = 1.6
    failure_counter: int = 0  # 记录连续失败和成功的次数。
    failure_tolerance: int = 5  # Note: Post-initialized 失败和成功的容忍度。当达到这些值时，会触发信赖域的收缩或扩展。
    success_counter: int = 0
    success_tolerance: int = 8  # Note: The original paper uses 3
    best_value: float = -float("inf")  # 记录到目前为止找到的最佳函数值。
    restart_triggered: bool = False  # 标记是否需要重启算法

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
    #     )


def update_state(state, Y_next):
    # 如果新的最佳值比当前最佳值提高了超过 0.1%，视为成功。否则视为失败。
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    # 如果连续成功次数达到 success_tolerance，信赖域扩大一倍（但不超过 length_max）。
    # 如果连续失败次数达到 failure_tolerance，信赖域缩小一半。
    # 如果信赖域长度小于 length_min，触发重启。
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def normalize(self, x, bounds=None):
    if bounds is None:
        bounds = self.bounds
    lb = bounds[:, 0]
    ub = bounds[:, 1]

    if x.ndimension() == 2 and x.shape[1] == 1:
        return (x - lb.view(-1, 1)) / (ub.view(-1, 1) - lb.view(-1, 1))
    else:
        return (x - lb) / (ub - lb)

def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=4096,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    # TS中选择候选点的数量
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # 中心点 x_center 选择为当前最佳点
    # 使用模型的长度尺度来缩放信赖域，这是一个很巧妙的设计，使得信赖域在不同维度上的大小与特征的重要性相适应
    # 计算信赖域的上下界 tr_lb 和 tr_ub
    # Scale the TR to be proportional to the lengthscales
    # x_center = X[Y.argmax(), :].clone()
    # base_kernel = model.covar_module.base_kernel
    # if hasattr(base_kernel, "lengthscale") and base_kernel.lengthscale is not None:
    #     lengthscale = base_kernel.lengthscale.squeeze().detach()
    # else:
    #     lengthscale = torch.ones(X.shape[-1], device=X.device)
    #
    # if hasattr(base_kernel, "alpha_phys") and hasattr(base_kernel, "alpha_rbf"):
    #     alpha_phys = base_kernel.alpha_phys
    #     alpha_rbf = base_kernel.alpha_rbf
    # else:
    #     alpha_phys = 0.7
    #     alpha_rbf = 0.3
    #
    # weights = torch.ones_like(lengthscale)
    #
    # indices_phys = base_kernel.indices_phys if hasattr(base_kernel, "indices_phys") else []
    # indices_rbf = base_kernel.indices_rbf if hasattr(base_kernel, "indices_rbf") else []
    #
    # if len(indices_phys) > 0:
    #     phys_weight_val = alpha_phys / len(indices_phys)
    #     weights[indices_phys] = phys_weight_val
    #
    # if len(indices_rbf) > 0:
    #     rbf_weight_val = alpha_rbf / len(indices_rbf)
    #     weights[indices_rbf] = rbf_weight_val
    #
    # weights = weights / weights.mean()
    # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    # tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    # tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    x_center = X[Y.argmax(), :].clone()
    base_kernel = model.covar_module.base_kernel

    if hasattr(base_kernel, "lengthscale") and base_kernel.lengthscale is not None:
        lengthscale = base_kernel.lengthscale.squeeze().detach()
    else:
        lengthscale = torch.ones(X.shape[-1], device=X.device)

    alpha_phys = getattr(base_kernel, "alpha_phys", torch.tensor(0.7, device=X.device))
    alpha_rbf = getattr(base_kernel, "alpha_rbf", torch.tensor(0.3, device=X.device))
    indices_phys = getattr(base_kernel, "indices_phys", [])
    indices_rbf = getattr(base_kernel, "indices_rbf", [])
    calu_func = getattr(base_kernel, "calu_func", None)

    weights = torch.ones_like(lengthscale)
    if len(indices_phys) > 0:
        weights[indices_phys] = alpha_phys / len(indices_phys)
    if len(indices_rbf) > 0:
        weights[indices_rbf] = alpha_rbf / len(indices_rbf)

    if calu_func is not None:
        x_samples = x_center + 0.05 * torch.randn(16, x_center.numel(), device=X.device)
        x_samples = x_samples.clamp(0, 1)
        f_center = calu_func(x_center.unsqueeze(0))
        f_samples = calu_func(x_samples)
        d_phys = torch.cdist(f_center, f_samples).mean()
        d_param = torch.norm(x_samples - x_center, dim=1).mean()
        if d_param > 1e-9:
            scale_ratio = (d_phys / d_param).detach().clamp(0.1, 10.0)
            weights = weights * scale_ratio

    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # 信赖域的上下界 tr_lb 和 tr_ub 使用 torch.clamp(...) 确保在 [0, 1] 范围内：

    # 使用 Sobol 序列在信赖域内生成候选点
    # 创建扰动掩码，控制哪些维度被扰动，这有助于在高维空间中进行有效探索
    # 基于掩码和扰动生成候选点集 X_cand
    # 使用 MaxPosteriorSampling 从候选点中选择下一批点
    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # 使用 Sobol 序列生成均匀分布在 [0, 1] 范围内的点。
        # 然后将这些点映射到信赖域 [tr_lb, tr_ub] 内。
        # 这确保了初始扰动均匀覆盖整个信赖域。

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1
        # prob_perturb 定义了每个维度被扰动的概率，最大为 1，最小为 20/dim。
        # 创建一个随机掩码，其中 True 表示该维度将被扰动。
        # 确保每个候选点至少有一个维度被扰动（处理全 False 的情况）。
        # 这个设计非常适合高维问题，因为它允许只改变部分维度，而不是所有维度。

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

        # MaxPosteriorSampling 是 BoTorch 库中实现 Thompson 采样的一个类。它的工作原理如下：
        # a) 从 GP 模型后验采样：它首先从高斯过程模型的后验分布中采样一个函数。
        # b) 评估候选点：对所有候选点使用这个采样的函数进行评估。
        # c) 选择最佳点：从这些评估结果中选择具有最高值的点。
        # d) 重复过程：如果需要多个样本（在这里是 batch_size 个），它会重复这个过程，每次都从 GP 后验中重新采样一个新的函数。
        # e) 无替换采样：replacement=False 参数确保在一个批次中不会重复选择同一个点。
    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

class CombinedKernel(Kernel):
    def __init__(
        self,
        alpha_phys=0.7,
        alpha_rbf=0.3,
        indices_phys=None,
        indices_rbf=None,
        calu_func=None,
        lengthscale=1.0,
    ):
        super().__init__(has_lengthscale=False)
        self.log_alpha_phys = Parameter(torch.log(torch.tensor(alpha_phys, dtype=torch.float64)))
        self.log_alpha_rbf = Parameter(torch.log(torch.tensor(alpha_rbf, dtype=torch.float64)))
        self.log_lengthscale = Parameter(torch.log(torch.tensor(lengthscale, dtype=torch.float64)))

        self.calu_func = calu_func

        self.indices_phys = indices_phys or []
        self.indices_rbf = indices_rbf or []

    @property
    def alpha_phys(self):
        return torch.exp(self.log_alpha_phys)

    @property
    def alpha_rbf(self):
        return torch.exp(self.log_alpha_rbf)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    def forward(self, x1, x2, diag=False, **params):
        if self.indices_rbf:
            x1_rbf = x1[:, self.indices_rbf]
            x2_rbf = x2[:, self.indices_rbf]
        else:
            x1_rbf = x1
            x2_rbf = x2

        diff_rbf = x1_rbf[:, None, :] - x2_rbf[None, :, :]
        K_rbf = torch.exp(-torch.norm(diff_rbf, dim=2) ** 2)

        if self.calu_func is not None:
            f1 = self.calu_func(x1)  # shape: (n1, d_phys)
            f2 = self.calu_func(x2)  # shape: (n2, d_phys)
            D_phys = torch.cdist(f1, f2, p=2)  # pairwise L2
            K_phys = torch.exp(-0.5 * (D_phys / self.lengthscale) ** 2)
        else:
            K_phys = torch.zeros_like(K_rbf)

        K_combined = self.alpha_rbf * K_rbf + self.alpha_phys * K_phys
        return K_combined


def turbo_solver(evaluate_func, bound, N_INIT=128, BATCH_SIZE=30, NUM_RESTARTS=10,RAW_SAMPLES=512,N_CANDIDATES=4096,acqf="ts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_cholesky_size = float("inf")
    DIM = len(bound[-1])
    dtype = torch.float64
    # bounds_tensor = torch.tensor(bound)
    bounds_tensor = torch.tensor(bound, device=device, dtype=dtype)

    X_turbo = get_initial_points(DIM, N_INIT).to(device=device, dtype=dtype)  # 已归一化到0，1的输入，tensor
    print(X_turbo)
    X_np = process_X(X_turbo, bounds_tensor)
    print(X_np)
    Y_turbo = evaluate_func(X_np)
    Y_turbo = torch.tensor(Y_turbo, device=device, dtype=dtype)
    print(Y_turbo)
    stats = calculate_iqr_stats(Y_turbo)
    print(f"Q1: {stats['Q1']}")
    print(f"Q2 (Median): {stats['Q2']}")
    print(f"Q3: {stats['Q3']}")
    print(f"IQR: {stats['IQR']}")
    threshold = stats['Q1']
    X_turbo, Y_turbo = filter_by_y_threshold(X_turbo, Y_turbo, threshold)
    X_turbo = X_turbo.to(device=device, dtype=dtype)
    Y_turbo = Y_turbo.to(device=device, dtype=dtype)
    print(f"Best initial point: {Y_turbo.min().item():.3f}")

    Y_turbo = -1 * torch.log(Y_turbo)
    state = TurboState(DIM, batch_size=BATCH_SIZE, best_value=max(Y_turbo).item())
    print(state)
    history = []
    iteration = 0
    while not state.restart_triggered:  # Run until TuRBO converges
        iteration += 1

        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        train_Y = train_Y.to(device=device, dtype=dtype)# 手动标准化
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)).to(device=device, dtype=dtype)
        # covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        #     MaternKernel(
        #         nu=2.5, ard_num_dims=DIM, lengthscale_constraint=Interval(0.005, 4.0)
        #     )
        # )
        # model = SingleTaskGP(
        #     X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        # )

        covar_module = ScaleKernel(
            CombinedKernel(
                alpha_phys=0.7,
                alpha_rbf=0.3,
                calu_func=calu,
                lengthscale=0.5,
                indices_rbf=list(range(0, DIM)),
            )
        ).to(device=device, dtype=dtype)

        model = SingleTaskGP(
            X_turbo,
            train_Y,
            covar_module=covar_module,
            likelihood=likelihood
        ).to(device=device, dtype=dtype)

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device=device, dtype=dtype)
        print(X_turbo.device)
        print(train_Y.device)
        print(Y_turbo.device)
        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            with debug(True):
                fit_gpytorch_mll(mll)

            # Create a batch 也是张量，在01之间
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=BATCH_SIZE,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf=acqf,
            )

        Y_next = evaluate_func(process_X(X_next,bounds_tensor))
        Y_next = -1 * torch.log(Y_next)
        Y_next = Y_next.to(device=device, dtype=dtype)
        # Update state
        state = update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # 还原最佳值
        original_best_value = math.exp(-state.best_value)

        # Print current status
        print(
            f"{iteration}) Best value: {original_best_value:.6f}, TR length: {state.length:.6f}"
        )
        history.append({
            "iteration": iteration,
            "best_value": original_best_value,
            "tr_length": state.length,
            "sigma": state.sigma if hasattr(state, "sigma") else None,
        })
    with open("turbo_optimization_history01.pkl", "wb") as f:
        pickle.dump(history, f)