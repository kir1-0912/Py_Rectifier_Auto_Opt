import numpy as np
from sko.GA import GA
from sko.tools import set_run_mode
import pandas as pd
import matplotlib.pyplot as plt
from Balun_X_Example_Single_Gen import Balun_X_Gen
from Allocate_EMX import Rect_Opt_Flow
from decimal import Decimal, ROUND_HALF_UP

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
        return Rect_Opt_Flow(x_norm)


set_run_mode(Object,'multiprocessing')



# with open('result.csv' ,'w') as f:
#     f.write(f"Fwn, Fwp, Fn_p, Lp, Mul_p, Fn_n, Ln, Mul_n, Cp, Cs, indp, inds, K, RL_2_4G , best_eta_2_4G, RL_5G, best_eta_5G , FoM\n")

with open('result.csv' ,'w') as f:
    f.write(f"Fwn, Fwp, Fn_p, Lp, Mul_p, Fn_n, Ln, Mul_n, Cp, Cs, indp, inds, K, RL_2_4G , best_eta_2_4G, FoM\n")



ga = GA(func=Object, n_dim=13, size_pop= 200, max_iter= 1000, prob_mut=0.3,
        lb=bound[0],
        ub=bound[1],
        precision=1e-3
        )

best_x, best_y =ga.run()

print("best_x: ",best_x,"\n","best_y: ",best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
Y_history.to_excel("history.xlsx")
fig, ax =plt.subplots(2,1)
ax[0].plot(Y_history.index,Y_history.values,'.',color='red')
Y_history.min(axis=1).cummin().plot(kind='line')

plt.show()