# Equations
# y = α1x1 + α2x2 + α3x3 + α0
# y = β1x1 + β2x2^2 + β3x3^3 + β0
# y = γ1x1 + γ2x2^2 + γ3x3 + γ0
# y = β1x1 + β2x1x2 + β0 # Q3 a

# formula --> J(a1, a2, a3, a0) = sum((y-yi)**2)
# https://docs.sympy.org/latest/modules/physics/vector/api/classes.html#sympy.physics.vector.vector.Vector.diff
import math
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import time

filename = "adj_real_estate.csv"

def diff_j_x(j, v):
    # partial derivative of equation J wrt variable v
    return sym.diff(j, v)

def answer_b_i(df, i):
    # Answer for Q3 b i
    print(f'***** {i} *****')
    b1, b2, b3, b0 = sym.symbols('b1 b2 b3 b0')
    y_y_cap_1, y_y_cap_2, y_y_cap_3 = [], [], []
    # y_y_cap_4 = [] # Q3 a

    # start_time = time.time()
    for r in range(len(df)):
        x1 = df['x1'][r]
        x2 = df['x2'][r]
        x3 = df['x3'][r]
        y = df['y'][r]
        y_y_cap_1.append(sym.Eq((y - x1*b1 - x2*b2 - x3*b3 - b0)**2, 0))
        y_y_cap_2.append(sym.Eq((y - x1*b1 - x2**2*b2 - x3**3*b3 - b0)**2, 0))
        y_y_cap_3.append(sym.Eq((y - x1*b1 - x2**2*b2 - x3*b3 - b0)**2, 0))
        # y_y_cap_4.append(sym.Eq((y - b1*x1 - b2*x1*x2 - b0)**2, 0)) # Q3 a

    # print('**** START DONE ****', time.time() - start_time)
    J1 = y_y_cap_1[0]
    J2 = y_y_cap_2[0]
    J3 = y_y_cap_3[0]
    # J4 = y_y_cap_4[0] # Q3 a

    # start_time = time.time()
    for i in range(1, len(y_y_cap_1)):
        J1 = sym.Eq(sym.simplify(J1.lhs + y_y_cap_1[i].lhs))
    for i in range(1, len(y_y_cap_2)):
        J2 = sym.Eq(sym.simplify(J2.lhs + y_y_cap_2[i].lhs))
    for i in range(1, len(y_y_cap_3)):
        J3 = sym.Eq(sym.simplify(J3.lhs + y_y_cap_3[i].lhs))
    # for i in range(1, len(y_y_cap_4)): # Q3 a
    #     J4 = sym.Eq(sym.simplify(J4.lhs + y_y_cap_4[i].lhs)) # Q3 a

    # print('**** JDONE ****', time.time() - start_time)
    # start_time = time.time()
    # print('J1:', sym.simplify(J1.lhs))
    eq1_J1 = sym.diff(J1.lhs, b1)
    eq2_J1 = sym.diff(J1.lhs, b2)
    eq3_J1 = sym.diff(J1.lhs, b3)
    eq0_J1 = sym.diff(J1.lhs, b0)
    sol_J1 = sym.solve([eq1_J1, eq2_J1, eq3_J1, eq0_J1], b1, b2, b3, b0)
    # print('**** J1 ****', time.time() - start_time)
    print("αs::\n α0-", sol_J1[b0], ", α1-", sol_J1[b1], ", α2-", sol_J1[b2], ", α3-", sol_J1[b3] )

    # start_time = time.time()
    # print('J2:', sym.simplify(J2.lhs))
    eq1_J2 = sym.diff(J2.lhs, b1)
    eq2_J2 = sym.diff(J2.lhs, b2)
    eq3_J2 = sym.diff(J2.lhs, b3)
    eq0_J2 = sym.diff(J2.lhs, b0)
    sol_J2 = sym.solve([eq1_J2, eq2_J2, eq3_J2, eq0_J2], b0, b1, b2, b3)
    # print('**** J2 ****', time.time() - start_time)
    print("βs::\n β0-", sol_J2[b0], ", β1-", sol_J2[b1], ", β2-", sol_J2[b2], ", β3-", sol_J2[b3] )

    # start_time = time.time()
    # print('J3:', sym.simplify(J3.lhs))
    eq1_J3 = sym.diff(J3.lhs, b1)
    eq2_J3 = sym.diff(J3.lhs, b2)
    eq3_J3 = sym.diff(J3.lhs, b3)
    eq0_J3 = sym.diff(J3.lhs, b0)
    sol_J3 = sym.solve([eq1_J3, eq2_J3, eq3_J3, eq0_J3], b0, b1, b2, b3)
    # print('**** J3 ****', time.time() - start_time)
    print("γs::\n γ0-", sol_J3[b0], ", γ1-", sol_J3[b1], ", γ2-", sol_J3[b2], ", γ3-", sol_J3[b3] )

    # start_time = time.time() # Q3 a
    # print('J4:', sym.simplify(J4.lhs)) # Q3 a
    # eq1_J4 = sym.diff(J4.lhs, b1) # Q3 a
    # eq2_J4 = sym.diff(J4.lhs, b2) # Q3 a
    # eq3_J4 = sym.diff(J4.lhs, b3) # Q3 a
    # eq0_J4 = sym.diff(J4.lhs, b0) # Q3 a
    # sol_J4 = sym.solve([eq1_J4, eq2_J4, eq3_J4, eq0_J4], b0, b1, b2, b3) # Q3 a
    # # print('**** J4 ****', time.time() - start_time) # Q3 a
    # print(sol_J4) # Q3 a

    Js = {'J1': J1, 'J2': J2, 'J3': J3, 'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'sol_J1': sol_J1, 'sol_J2': sol_J2, 'sol_J3': sol_J3} # , 'J4': J4, 'sol_J4': sol_J4
    return Js

def answer_b_ii(df):
    # Answer for Q3 b ii
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
    from sklearn.model_selection import LeaveOneOut
    x, y = df[['x1', 'x2', 'x3']], df['y']
    cv = LeaveOneOut()
    i = 0
    start_time = time.time()
    rmse_all = []
    for trn, tst in cv.split(x):
        i += 1
        x_train, x_test = x.iloc[list(trn), :], x.iloc[list(tst), :]
        y_train, y_test = y.iloc[list(trn)], y.iloc[list(tst)]
        train_df = x_train.assign(y=y_train).reset_index()
        Js = answer_b_i(train_df, i)
        test_df = x_test.assign(y=y_test).reset_index()
        rmse_all.append(predict_y(Js, test_df, i))
        print("(", i, ": Time taken:", time.time() - start_time, ")")
    print('RMSE J1=', np.mean([j['rmse_J1'] for j in rmse_all]))
    print('RMSE J2=', np.mean([j['rmse_J2'] for j in rmse_all]))
    print('RMSE J3=', np.mean([j['rmse_J3'] for j in rmse_all]))

def predict_y(Js, test_df, i):
    sol_J1, sol_J2, sol_J3 = Js['sol_J1'], Js['sol_J2'], Js['sol_J3']
    # sol_J4 = Js['sol_J4']
    b0, b1, b2, b3 = Js['b0'], Js['b1'], Js['b2'], Js['b3']
    print(f'### Prediction for {i} ###')
    x1 = test_df['x1']
    x2 = test_df['x2']
    x3 = test_df['x3']
    y = test_df['y']
    y_pred_J1 = x1 * sol_J1[b1] + x2 * sol_J1[b2] + x3 * sol_J1[b3] + sol_J1[b0]
    y_pred_J2 = x1 * sol_J2[b1] + x2**2 * sol_J2[b2] + x3**3 * sol_J2[b3] + sol_J2[b0]
    y_pred_J3 = x1 * sol_J3[b1] + x2**2 * sol_J3[b2] + x3 * sol_J3[b3] + sol_J3[b0]
    rmse_J1 = y_pred_J1 - y # RMSE for LOOCV is sqrt(sqr(mean(single_number))) = single_number
    rmse_J2 = y_pred_J2 - y
    rmse_J3 = y_pred_J3 - y
    print(f'RMSE for J1: {rmse_J1}, J2: {rmse_J2}, J3: {rmse_J3}')
    rmse = { 'rmse_J1': rmse_J1, 'rmse_J2': rmse_J2, 'rmse_J3': rmse_J3 }
    return rmse

if __name__=="__main__":
    print('Equations are: \ny = α1x1 + α2x2 + α3x3 + α0\ny = β1x1 + β2x2^2 + β3x3^3 + β0\ny = γ1x1 + γ2x2^2 + γ3x3 + γ0')
    dataset_df = pd.read_csv(filename)
    dataset_df.rename(columns={'X1 house age': 'x1', 'X2 distance to the nearest MRT station': 'x2', 'X3 number of convenience stores': 'x3', 'Y house price of unit area': 'y'}, inplace=True)
    # dataset_df = pd.DataFrame(columns=['x1','x2','x3','y'], data=[[146.118721,2.616478,200,64.510638], [89.041096,9.451101,180,71.829787], [60.730594,17.323757,100,80.510638], [22.831050,12.039677,100,73.361702]]) # Test
    # dataset_df = pd.DataFrame(columns=['x1','x2','y'], data=[[1,2,3], [0,5,2], [2,2,4], [3,1,0]]) # Q3 a
    print("\n----- 3 b i - GLR - Coeff -----")
    total_time_b_i = time.time()
    Js = answer_b_i(dataset_df, i=0)
    # print('$$$$ end answer 2 b i', time.time() - total_time_b_i, '$$$$')
    print("\n----- 3 b ii - GLR - LOOCV -----")
    total_time_b_ii = time.time()
    # answer_b_ii(dataset_df)
    # print('$$$$ end answer 2 b ii', time.time() - total_time_b_ii, '$$$$')

##### ANSWERS #####
"""
----- 3 b i - GLR - Coeff -----
αs::
 α0- 73.1528275854021 , α1- -0.0942560442848780 , α2- -0.297020476292674 , α3- 0.110420636248531
βs::
 β0- 71.0848687172470 , β1- -0.0981564170878302 , β2- -0.00170695120488303 , β3- 4.18210141778758e-6
γs::
 γ0- 62.2959766631508 , γ1- -0.0959114064346790 , γ2- -0.00134455292576147 , γ3- 0.166573210224225
----- 3 b ii - GLR - LOOCV -----
RMSE J1= -0.02697056637387293
RMSE J2= -0.02843872019662596
RMSE J3= -0.036311504548677
"""