import numpy as np
import multiprocessing as mp
import torch
from torch.quasirandom import SobolEngine
from scipy.stats import qmc
from botorch.models import ModelListGP, MultiTaskGP
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import DeterministicModel, GenericDeterministicModel
from botorch.models.model import Model
import time
from botorch.utils.gp_sampling import (
    RandomFourierFeatures,
    get_weights_posterior,
    get_deterministic_model,
    get_deterministic_model_multi_samples,
)
import time
import os
import numpy as np




tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

def normalize(X, bounds_tensor):
    """
    将 X 从原始范围归一化到 [0, 1] 范围。

    参数:
    X : torch.Tensor, 形状为 (n, DIM)
    bounds_tensor : torch.Tensor, 形状为 (2, DIM)，第一行是下界，第二行是上界

    返回:
    X_normalized : torch.Tensor, 形状为 (n, DIM)，值在 [0, 1] 范围内
    """
    lower_bounds = bounds_tensor[0]
    upper_bounds = bounds_tensor[1]

    X_normalized = (X - lower_bounds) / (upper_bounds - lower_bounds)
    return X_normalized

def denormalize(X_normalized, bounds_array, shift=0):
    """
    将 X_normalized 从 [0+shift, 1+shift] 范围反归一化到原始范围。

    参数:
    X_normalized : numpy.ndarray, 形状为 (n, DIM)，值在 [0+shift, 1+shift] 范围内
    bounds_array : numpy.ndarray, 形状为 (2, DIM)，第一行是下界，第二行是上界
    shift : float, 默认为0，用于调整输入范围

    返回:
    X : numpy.ndarray, 形状为 (n, DIM)，值在原始范围内
    """
    device = X_normalized.device
    bounds_array = bounds_array.to(device)
    lower_bounds = bounds_array[0]
    upper_bounds = bounds_array[1]

    X = (X_normalized + shift) * (upper_bounds - lower_bounds) + lower_bounds
    return X














def process_X(X_tensor, bounds_tensor):
    X = denormalize(X_tensor, bounds_tensor)
    X_np = X.detach().cpu().numpy()
    return X_np

seed=int(time.time())

# 生成(0,1)的初始点
def get_initial_points(dim, n_pts, seed=seed):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(**tkwargs)
    return X_init


def filter_by_y_threshold(X, Y, threshold):
    """
    过滤 X 和 Y，只保留 Y 小于阈值的部分。
    """
    # 确保 Y 是 2D tensor
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    # 创建掩码
    mask = Y.squeeze() < threshold

    # 应用掩码
    filtered_X = X[mask]
    filtered_Y = Y[mask]

    return filtered_X, filtered_Y


def calculate_iqr_stats(Y):
    # 确保Y是PyTorch张量
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)

    # 确保Y是一维张量
    Y = Y.flatten()

    # 计算Q1, Q2, Q3
    q1 = Y.quantile(0.25)
    q2 = Y.median()
    q3 = Y.quantile(0.75)

    # 计算IQR
    iqr = q3 - q1

    return {
        "Q1": q1.item(),
        "Q2": q2.item(),
        "Q3": q3.item(),
        "IQR": iqr.item()
    }


def create_initial_data(evaluate_func, bound, n_init=256):
    # 创建初始训练集，进行256次仿真然后取前1/4好的

    bounds_tensor = torch.tensor(bound)
    lower, upper = bounds_tensor[0], bounds_tensor[1]
    sampler = qmc.Sobol(d=len(bound[-1]), scramble=True)
    sample = sampler.random(n=n_init)
    X_np = qmc.scale(sample, lower, upper)
    X = torch.from_numpy(X_np).to(**tkwargs)
    Y = evaluate_func(X_np)
    stats = calculate_iqr_stats(Y)
    print(f"Q1: {stats['Q1']}")
    print(f"Q2 (Median): {stats['Q2']}")
    print(f"Q3: {stats['Q3']}")
    print(f"IQR: {stats['IQR']}")

    threshold = stats['Q1']
    X, Y = filter_by_y_threshold(X, Y, threshold)

    print(f"Best initial point: {Y.min().item():.3f}")

    return X, Y


def get_gp_samples(
    model: Model,
    num_outputs: int,
    n_samples: int,
    num_rff_features: int = 512,
) -> GenericDeterministicModel:
    r"""Sample functions from GP posterior using RFFs. The returned
    `GenericDeterministicModel` effectively wraps `num_outputs` models,
    each of which has a batch shape of `n_samples`. Refer
    `get_deterministic_model_multi_samples` for more details.

    Args:
        model: The model.
        num_outputs: The number of outputs.
        n_samples: The number of functions to be sampled IID. 要采样的独立同分布（IID）函数数量
        num_rff_features: The number of random Fourier features.

    Returns:
        A batched `GenericDeterministicModel` that batch evaluates `n_samples`
        sampled functions.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
        else:
            models = model.models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    weights = []
    bases = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        train_targets = models[m].train_targets
        # get random fourier features
        # sample_shape controls the number of iid functions.
        basis = RandomFourierFeatures(
            kernel=models[m].covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
            sample_shape=torch.Size([n_samples] if n_samples > 1 else []),
        )
        bases.append(basis)
        # TODO: when batched kernels are supported in RandomFourierFeatures,
        # the following code can be uncommented.
        # if train_X.ndim > 2:
        #    batch_shape_train_X = train_X.shape[:-2]
        #    dataset_shape = train_X.shape[-2:]
        #    train_X = train_X.unsqueeze(-3).expand(
        #        *batch_shape_train_X, n_samples, *dataset_shape
        #    )
        #    train_targets = train_targets.unsqueeze(-2).expand(
        #        *batch_shape_train_X, n_samples, dataset_shape[0]
        #    )
        phi_X = basis(train_X)
        # Sample weights from bayesian linear model
        # 1. When inputs are not batched, train_X.shape == (n, d)
        # weights.sample().shape == (n_samples, num_rff_features)
        # 2. When inputs are batched, train_X.shape == (batch_shape_input, n, d)
        # This is expanded to (batch_shape_input, n_samples, n, d)
        # to maintain compatibility with RFF forward semantics
        # weights.sample().shape == (batch_shape_input, n_samples, num_rff_features)
        mvn = get_weights_posterior(
            X=phi_X,
            y=train_targets,
            sigma_sq=models[m].likelihood.noise.mean().item(),
        )
        weights.append(mvn.sample())

    # TODO: Ideally support RFFs for multi-outputs instead of having to
    # generate a basis for each output serially.
    if n_samples > 1:
        return get_deterministic_model_multi_samples(
            weights=weights,
            bases=bases,
        )
    return get_deterministic_model(
        weights=weights,
        bases=bases,
    )


def get_gp_sample_w_transforms(
    model: Model,
    num_outputs: int,
    n_samples: int,
    num_rff_features: int = 512,
) -> DeterministicModel:
    intf = None
    octf = None
    if hasattr(model, "input_transform"):
        intf = model.input_transform
    if hasattr(model, "outcome_transform"):
        octf = model.outcome_transform
        model.outcome_transform = None
    base_gp_samples = get_gp_samples(
        model=model,
        num_outputs=num_outputs,
        n_samples=n_samples,
        num_rff_features=num_rff_features,
    )
    if intf is not None:
        base_gp_samples.input_transform = intf
        model.input_transform = intf
    if octf is not None:
        base_gp_samples.outcome_transform = octf
        model.outcome_transform = octf
    return base_gp_samples




