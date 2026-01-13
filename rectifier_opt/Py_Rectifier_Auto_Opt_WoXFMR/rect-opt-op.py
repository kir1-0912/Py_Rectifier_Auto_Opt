import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import pandas as pd
import matplotlib.pyplot as plt
from Balun_X_Example_Single_Gen import Balun_X_Gen
from Allocate_EMX import Rect_Opt_Flow

bound_dict = {
    "FWn": [10, 300],  # 10*10n 300*10n
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

space = [
    Real(bound_dict["FWn"][0], bound_dict["FWn"][1]),
    Real(bound_dict["Fwp"][0], bound_dict["Fwp"][1]),
    Real(bound_dict["Fn_p"][0], bound_dict["Fn_p"][1]),
    Real(bound_dict["Lp"][0], bound_dict["Lp"][1]),
    Real(bound_dict["Mul_p"][0], bound_dict["Mul_p"][1]),
    Real(bound_dict["Fn_n"][0], bound_dict["Fn_n"][1]),
    Real(bound_dict["Ln"][0], bound_dict["Ln"][1]),
    Real(bound_dict["Mul_n"][0], bound_dict["Mul_n"][1]),
    Real(bound_dict["Cp"][0], bound_dict["Cp"][1]),
    Real(bound_dict["Cs"][0], bound_dict["Cs"][1]),
    Real(bound_dict["indp"][0], bound_dict["indp"][1]),
    Real(bound_dict["inds"][0], bound_dict["inds"][1]),
    Real(bound_dict["K"][0], bound_dict["K"][1])
]

def objective(x):
    return Rect_Opt_Flow(x)

result = gp_minimize(objective, space, n_calls=10, random_state=0)

print("Best parameters found: ", result.x)
print("Function value at best parameters: ", result.fun)

Y_history = pd.DataFrame(result.func_vals)
Y_history.to_excel("history.xlsx")

fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')

plt.show()