import time
import numpy as np
from Allocate_EMX import Rect_Opt_Flow
import torch



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
                 , dtype=np.float64)
bounds = bound.T
param_names = list(bound_dict.keys())
n_params = len(param_names)
int_param_indices = list(range(10))

def normalize(x, bounds):
    x = np.asarray(x, dtype=np.float64)
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def denormalize(x_norm, bounds):
    x_norm = np.asarray(x_norm, dtype=np.float64)
    return bounds[:, 0] + x_norm * (bounds[:, 1] - bounds[:, 0])

def simulate(X_np):
    result = Rect_Opt_Flow(X_np)
    y = np.log10(result)
    return np.float64(y)


def compute_gradient(x_norm, bounds, h=1e-3):
    grad = np.zeros_like(x_norm, dtype=np.float64)
    for i in range(len(x_norm)):
        delta = np.float64(h) if i not in int_param_indices else np.float64(1.0 / (bounds[i, 1] - bounds[i, 0]))
        x1 = x_norm.copy()
        x2 = x_norm.copy()
        x1[i] = min(1.0, x1[i] + delta)
        x2[i] = max(0.0, x2[i] - delta)
        p1 = simulate(denormalize(x1, bounds))
        p2 = simulate(denormalize(x2, bounds))
        grad[i] = (p1 - p2) / (2 * delta)
    return grad

def optimize(bounds, alpha_set=[0.5, 1, 1.5, 2], h=1e-3, epsilon_eta=1e-4, epsilon_alpha=1e-4):
    x_norm = np.full(bound.shape[1], 0.7, dtype=np.float64)
    t1 = time.time()
    best_pce = simulate(denormalize(x_norm, bounds))
    t2 = time.time()
    print(f"Initial PCE: {best_pce:.4f}")
    print(f"sim time", (t2-t1))
    while True:
        t3 = time.time()
        grad = compute_gradient(x_norm, bounds, h=h)
        t4 = time.time()
        print(f"sim time13", (t4-t3))
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            print("Gradient too small, convergence.")
            break
        direction = grad / grad_norm
        print(direction)
        improved = False
        best_step = None
        best_candidate = None
        best_new_pce = best_pce

        for alpha in alpha_set:
            candidate = x_norm + alpha * direction
            print(candidate)
            candidate = np.clip(candidate, 0, 1)
            candidate_pce = simulate(denormalize(candidate, bounds))
            if candidate_pce > best_new_pce:
                best_new_pce = candidate_pce
                best_step = alpha
                best_candidate = candidate
                improved = True

        if not improved:
            alpha_set = [a / 2 for a in alpha_set if a / 2 >= epsilon_alpha]
            print(alpha_set)
            if not alpha_set:
                print("No further step improves PCE. Optimization finished.")
                break
            continue

        if best_new_pce - best_pce < epsilon_eta:
            print("PCE improvement too small. Optimization finished.")
            break

        x_norm = best_candidate
        best_pce = best_new_pce
        print(f"New PCE: {best_pce:.4f} at step ¦Á={best_step:.4f}")

    final_params = denormalize(x_norm, bounds)
    return dict(zip(param_names, final_params)), best_pce

param, best_pce=optimize(bounds)
print(param)
print(best_pce)
