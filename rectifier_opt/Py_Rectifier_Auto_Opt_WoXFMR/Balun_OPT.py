import numpy as np
from sko.GA import GA
from sko.tools import set_run_mode
import pandas as pd
import matplotlib.pyplot as plt
from Balun_X_Example_Single_Gen import Balun_X_Gen
from Allocate_EMX import Rect_Opt_Flow
from decimal import Decimal, ROUND_HALF_UP

## bound order: BalunLength trackWidth trackspacing primeTurns SecondaryTurns
bound = np.array([ [100,3,3,1,1]
                  ,[500,20,20,10,10]
                  ])

def Object(x_norm):
        # return -(sum(x_norm))
        x_norm[3:] = np.round(x_norm[3:]).astype(int)
        x_norm[0] = Decimal(x_norm[0]).quantize(Decimal('0.05'), rounding=ROUND_HALF_UP)
        x_norm[1] = Decimal(x_norm[1]).quantize(Decimal('0.05'), rounding=ROUND_HALF_UP)
        x_norm[2] = Decimal(x_norm[2]).quantize(Decimal('0.05'), rounding=ROUND_HALF_UP)
        # x_norm[3] = Decimal(x_norm[3]).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        # x_norm[4] = Decimal(x_norm[4]).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        return Rect_Opt_Flow(BalunLength=x_norm[0],
                              trackWidth=x_norm[1],
                              trackspacing=x_norm[2],
                              primaryTurns=x_norm[3],
                              SecondaryTurns=x_norm[4])
        # return OPAMP36_3(x_norm)


set_run_mode(Object,'multiprocessing')

with open('result.csv' ,'w') as f:
    f.write(f"CellName, Lp_2_4G, Qp_2_4G, Ls_2_4G, Qs_2_4G, K_2_4G, SRF_Lp, SRF_Ls, FoM \n")


ga = GA(func=Object, n_dim=5, size_pop= 60, max_iter= 100, prob_mut=0.1,
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