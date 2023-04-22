# Equations
# y = α1x1 + α2x2 + α3x3 + α0
# y = β1x1 + β2x2^2 + β3x3^3 + β0
# y = γ1x1 + γ2x2^2 + γ3x3 + γ0

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import time

filename = "adj_real_estate.csv"

def answer_b_i(df, i, print_flag=False):
    # Answer for Q3 b i
    print(f'***** Training {i} *****')
    coeff_1_J1, coeff_2_J1, coeff_3_J1, coeff_0_J1 = [], [], [], []
    eq0_J1, eq1_J1, eq2_J1, eq3_J1 = [], [], [], []
    y1_J1, y2_J1, y3_J1, y0_J1 = 0, 0, 0, 0

    coeff_1_J2, coeff_2_J2, coeff_3_J2, coeff_0_J2 = [], [], [], []
    eq0_J2, eq1_J2, eq2_J2, eq3_J2 = [], [], [], []
    y1_J2, y2_J2, y3_J2, y0_J2 = 0, 0, 0, 0

    coeff_1_J3, coeff_2_J3, coeff_3_J3, coeff_0_J3 = [], [], [], []
    eq0_J3, eq1_J3, eq2_J3, eq3_J3 = [], [], [], []
    y1_J3, y2_J3, y3_J3, y0_J3 = 0, 0, 0, 0

    for r in range(len(df)):
        x1 = df['x1'][r]
        x2 = df['x2'][r]
        x3 = df['x3'][r]
        y = df['y'][r]

        coeff_0_J1.append([1*1, 1*x1, 1*x2, 1*x3]) # d0*d0, d0*d1, d0*d2, d0*d3
        coeff_1_J1.append([x1*1, x1*x1, x1*x2, x1*x3]) # d1*d0, d1*d1, d1*d2, d1*d3
        coeff_2_J1.append([x2*1, x2*x1, x2*x2, x2*x3]) # d2*d0, d2*d1, d2*d2, d2*d3
        coeff_3_J1.append([x3*1, x3*x1, x3*x2, x3*x3]) # d3*d0, d3*d1, d3*d2, d3*d3
        y0_J1 += 1*y
        y1_J1 += x1*y
        y2_J1 += x2*y
        y3_J1 += x3*y

        coeff_0_J2.append([1*1, 1*x1, 1*x2**2, 1*x3**3]) # d0*d0, d0*d1, d0*d2, d0*d3
        coeff_1_J2.append([x1*1, x1*x1, x1*x2**2, x1*x3**3]) # d1*d0, d1*d1, d1*d2, d1*d3
        coeff_2_J2.append([x2**2*1, x2**2*x1, x2**2*x2**2, x2**2*x3**3]) # d2*d0, d2*d1, d2*d2, d2*d3
        coeff_3_J2.append([x3**3*1, x3**3*x1, x3**3*x2**2, x3**3*x3**3]) # d3*d0, d3*d1, d3*d2, d3*d3
        y0_J2 += 1*y
        y1_J2 += x1*y
        y2_J2 += x2**2*y
        y3_J2 += x3**3*y

        coeff_0_J3.append([1*1, 1*x1, 1*x2**2, 1*x3]) # d0*d0, d0*d1, d0*d2, d0*d3
        coeff_1_J3.append([x1*1, x1*x1, x1*x2**2, x1*x3]) # d1*d0, d1*d1, d1*d2, d1*d3
        coeff_2_J3.append([x2**2*1, x2**2*x1, x2**2*x2**2, x2**2*x3]) # d2*d0, d2*d1, d2*d2, d2*d3
        coeff_3_J3.append([x3*1, x3*x1, x3*x2**2, x3*x3]) # d3*d0, d3*d1, d3*d2, d3*d3
        y0_J3 += 1*y
        y1_J3 += x1*y
        y2_J3 += x2**2*y
        y3_J3 += x3*y

    for j in range(len(coeff_0_J1[0])):
        eq0_J1.append(sum([k[j] for k in coeff_0_J1]))
    for j in range(len(coeff_1_J1[0])):
        eq1_J1.append(sum([k[j] for k in coeff_1_J1]))
    for j in range(len(coeff_2_J1[0])):
        eq2_J1.append(sum([k[j] for k in coeff_2_J1]))
    for j in range(len(coeff_2_J1[0])):
        eq3_J1.append(sum([k[j] for k in coeff_3_J1]))
    n_J1 = len(eq0_J1)
    vars = solve_eq_matrix(n_J1, [eq0_J1, eq1_J1, eq2_J1, eq3_J1], [[y0_J1], [y1_J1], [y2_J1], [y3_J1]])
    sol_J1 = {'b0': vars[0][0], 'b1': vars[1][0], 'b2': vars[2][0], 'b3': vars[3][0]}
    if print_flag: print("αs::\n α0-", sol_J1['b0'], ", α1-", sol_J1['b1'], ", α2-", sol_J1['b2'], ", α3-", sol_J1['b3'])

    for j in range(len(coeff_0_J2[0])):
        eq0_J2.append(sum([k[j] for k in coeff_0_J2]))
    for j in range(len(coeff_1_J2[0])):
        eq1_J2.append(sum([k[j] for k in coeff_1_J2]))
    for j in range(len(coeff_2_J2[0])):
        eq2_J2.append(sum([k[j] for k in coeff_2_J2]))
    for j in range(len(coeff_2_J2[0])):
        eq3_J2.append(sum([k[j] for k in coeff_3_J2]))
    n_J2 = len(eq0_J2)
    vars = solve_eq_matrix(n_J2, [eq0_J2, eq1_J2, eq2_J2, eq3_J2], [[y0_J2], [y1_J2], [y2_J2], [y3_J2]])
    sol_J2 = {'b0': vars[0][0], 'b1': vars[1][0], 'b2': vars[2][0], 'b3': vars[3][0]}
    if print_flag: print("βs::\n β0-", sol_J2['b0'], ", β1-", sol_J2['b1'], ", β2-", sol_J2['b2'], ", β3-", sol_J2['b3'])

    for j in range(len(coeff_0_J3[0])):
        eq0_J3.append(sum([k[j] for k in coeff_0_J3]))
    for j in range(len(coeff_1_J3[0])):
        eq1_J3.append(sum([k[j] for k in coeff_1_J3]))
    for j in range(len(coeff_2_J3[0])):
        eq2_J3.append(sum([k[j] for k in coeff_2_J3]))
    for j in range(len(coeff_2_J3[0])):
        eq3_J3.append(sum([k[j] for k in coeff_3_J3]))
    n_J3 = len(eq0_J3)
    vars = solve_eq_matrix(n_J3, [eq0_J3, eq1_J3, eq2_J3, eq3_J3], [[y0_J3], [y1_J3], [y2_J3], [y3_J3]])
    sol_J3 = {'b0': vars[0][0], 'b1': vars[1][0], 'b2': vars[2][0], 'b3': vars[3][0]}
    if print_flag: print("γs::\n γ0-", sol_J3['b0'], ", γ1-", sol_J3['b1'], ", γ2-", sol_J3['b2'], ", γ3-", sol_J3['b3'])

    Js = {'sol_J1': sol_J1, 'sol_J2': sol_J2, 'sol_J3': sol_J3}
    return Js

def solve_eq_matrix(n, X, Y): # https://integratedmlai.com/system-of-equations-solution/
    idx = list(range(n))
    for d in range(n):
        d_scaler = 1.0 / X[d][d]
        for j in range(n):
            X[d][j] *= d_scaler
        Y[d][0] *= d_scaler
        for i in idx[0:d] + idx[d+1:]:
            cr_sclr = X[i][d]
            for j in range(n):
                X[i][j] = X[i][j] - cr_sclr * X[d][j]
            Y[i][0] = Y[i][0] - cr_sclr * Y[d][0]
    return Y

def answer_b_ii(df):
    # Answer for Q3 b ii
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
    from sklearn.model_selection import LeaveOneOut
    x, y = df[['x1', 'x2', 'x3']], df['y']
    cv = LeaveOneOut()
    i = 0
    rmse_all = []
    start_time = time.time()
    for trn, tst in cv.split(x):
        i += 1
        x_train, x_test = x.iloc[list(trn), :], x.iloc[list(tst), :]
        y_train, y_test = y.iloc[list(trn)], y.iloc[list(tst)]
        train_df = x_train.assign(y=y_train).reset_index()
        Js = answer_b_i(train_df, i, print_flag=False)
        test_df = x_test.assign(y=y_test).reset_index()
        rmse_all.append(predict_y(Js, test_df, i))
        print("(", i, ": Time elapsed:", time.time() - start_time, ")")
    print('RMSE J1=', np.sqrt(np.mean([j['rmse_J1'] for j in rmse_all])))
    print('RMSE J2=', np.sqrt(np.mean([j['rmse_J2'] for j in rmse_all])))
    print('RMSE J3=', np.sqrt(np.mean([j['rmse_J3'] for j in rmse_all])))

def predict_y(Js, test_df, i):
    sol_J1, sol_J2, sol_J3 = Js['sol_J1'], Js['sol_J2'], Js['sol_J3']
    print(f'### Evaluating {i} ###')
    x1 = test_df['x1']
    x2 = test_df['x2']
    x3 = test_df['x3']
    y = test_df['y']
    y_pred_J1 = x1 * sol_J1['b1'] + x2 * sol_J1['b2'] + x3 * sol_J1['b3'] + sol_J1['b0']
    y_pred_J2 = x1 * sol_J2['b1'] + x2**2 * sol_J2['b2'] + x3**3 * sol_J2['b3'] + sol_J2['b0']
    y_pred_J3 = x1 * sol_J3['b1'] + x2**2 * sol_J3['b2'] + x3 * sol_J3['b3'] + sol_J3['b0']
    rmse_J1 = (y_pred_J1 - y)**2 # RMSE for LOOCV is sqrt(sqr(mean(single_number))) = single_number
    rmse_J2 = (y_pred_J2 - y)**2
    rmse_J3 = (y_pred_J3 - y)**2
    rmse = { 'rmse_J1': rmse_J1, 'rmse_J2': rmse_J2, 'rmse_J3': rmse_J3 }
    return rmse

if __name__=="__main__":
    print('Equations are: \ny = α1x1 + α2x2 + α3x3 + α0\ny = β1x1 + β2x2^2 + β3x3^3 + β0\ny = γ1x1 + γ2x2^2 + γ3x3 + γ0')
    dataset_df = pd.read_csv(filename)
    dataset_df.rename(columns={'X1 house age': 'x1', 'X2 distance to the nearest MRT station': 'x2', 'X3 number of convenience stores': 'x3', 'Y house price of unit area': 'y'}, inplace=True)
    print("\n----- 3 b i - GLR - Coeff -----")
    total_time_b_i = time.time()
    Js = answer_b_i(dataset_df, i=0, print_flag=True)
    print('$$$$ ending answer 3b i in', time.time() - total_time_b_i, '$$$$')
    print("\n----- 3 b ii - GLR - LOOCV -----")
    total_time_b_ii = time.time()
    answer_b_ii(dataset_df)
    print('$$$$ ending answer 3b ii in', time.time() - total_time_b_ii, '$$$$')

##### ANSWERS #####
"""
----- 3 b i - GLR - Coeff -----
αs::
 α0- 73.15282758540117 , α1- -0.0942560442848722 , α2- -0.29702047629266937 , α3- 0.11042063624853207
βs::
 β0- 71.08486871724718 , β1- -0.09815641708783077 , β2- -0.0017069512048830343 , β3- 4.182101417787667e-06
γs::
 γ0- 62.295976663150185 , γ1- -0.0959114064346724 , γ2- -0.0013445529257614737 , γ3- 0.16657321022422367

----- 3 b ii - GLR - LOOCV -----
RMSE J1= 11.154092914152525
RMSE J2= 12.900771379471085
RMSE J3= 12.427370792893706
"""