import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Balun_X_Example_Single_Gen import Balun_X_Gen
from Allocate_EMX import Rect_Opt_Flow
from kboy import bo_solver
import pickle
import torch
import os

bound_dict={
    "FWn":              [10,300],   #10*10n 300*10n
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


def Object(x_norm):
    x_list = [item for sublist in x_norm for item in sublist]
    return Rect_Opt_Flow(x_list)


bo_solver(
    funct=Object,
    dim=bound.shape[1],
    bounds=bound.T,
    init_x=None,
    init_y=None,
    sigma=0.5,
    mu=0.5,
    c1=None,
    c2=None,
    allround_flag=False,
    greedy_flag=True,
    n_training=None,
    batch_size=10,
    n_candidates=200,
    n_resample=10,
    nMax=2000,
    k=13,
    dataset_file='boy.pkl',
    use_TS=False
)

# Load and process results
with open('boy.pkl', 'rb') as f:
    results = pickle.load(f)

x_samples = torch.stack(results['x'])
y_samples = torch.tensor(results['y'])



best_index = torch.argmin(y_samples)
best_x = x_samples[best_index]
best_y = y_samples[best_index]

print("Best fom:", best_y.item())
print("Best para:", best_x.tolist())


# plt.figure(figsize=(10, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(y_samples.numpy(), marker='o', label='Sampled Y')
# plt.title('Sampling Convergence')
# plt.xlabel('Iteration')
# plt.ylabel('Objective Value (f(x))')
# plt.grid(True)
# plt.legend()
#
#
# plt.subplot(1, 2, 2)
# best_so_far = torch.minimum.accumulate(y_samples)
# plt.plot(best_so_far.numpy(), color='orange', label='Best so far')
# plt.title('Best Objective Value vs Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Best f(x) so far')
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
#
# plt.show()
plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.plot(y_samples.numpy(), marker='o', label='Sampled Y')
plt.title('Sampling Convergence')
plt.xlabel('Iteration')
plt.ylabel('Objective Value (f(x))')
plt.grid(True)
plt.legend()


best_so_far, _ = torch.cummin(y_samples, dim=0)
plt.subplot(1, 2, 2)
plt.plot(best_so_far.numpy(), color='orange', label='Best so far')
plt.title('Best Objective Value vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Best f(x) so far')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


log_y_samples = torch.log10(y_samples)
plt.figure()
plt.plot(log_y_samples.numpy(), marker='x', label='log10(f(x))')
plt.title("Log-scaled Sampling Convergence")
plt.xlabel("Iteration")
plt.grid(True)
plt.legend()