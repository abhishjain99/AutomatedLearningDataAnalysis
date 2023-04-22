import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

filename = "svm_2022/svm_data_2022.csv"
train_file = "svm_2022/train_data_2022.csv"
test_file = "svm_2022/test_data_2022.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
attr = train_df.shape[1] - 1
X_train = train_df.iloc[:, 0:attr]
Y_train = train_df.iloc[:,attr]
X_test = test_df.iloc[:, 0:attr]
Y_test = test_df.iloc[:,attr]

def q5a_load_db(filename):
    print("----- (a) Data Loading -----")
    df = pd.read_csv(filename)
    print("Positive Samples=", len(df[df["Class"]==1]), "\nNegative Samples=", len(df[df["Class"]==0]))
    return df

def q5b_stratified_sampling(df, train_size=0.8):
    print("\n----- (b) Stratified Sampling -----")
    train_1, test_1 = np.split(df[df["Class"]==1].sample(frac=1), [int(train_size * len(df[df["Class"]==1]))])
    train_0, test_0 = np.split(df[df["Class"]==0].sample(frac=1), [int(train_size * len(df[df["Class"]==0]))])
    train_df_ss = pd.concat([train_0, train_1], ignore_index=True)
    test_df_ss = pd.concat([test_0, test_1], ignore_index=True)
    print("Training:: Positive Samples=", len(train_df_ss[train_df_ss["Class"]==1]), "; Negative Samples=", len(train_df_ss[train_df_ss["Class"]==0]))
    print("Testing :: Positive Samples=", len(test_df_ss[test_df_ss["Class"]==1]), "; Negative Samples=", len(test_df_ss[test_df_ss["Class"]==0]))

# SVM Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def q5c_svm_linear_kernel():
    print("\n----- (c) SVM with Linear Kernel -----")
    C = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10] # regularization parameter
    svs = []
    for i in C:
        print(f"***** C={i} *****")
        # Training
        svclass = SVC(kernel="linear", C=i)
        svclass.fit(X_train, Y_train)

        print(f"Support vectors for class {svclass.classes_[0]}: {svclass.n_support_[0]}")
        print(f"Support vectors for class {svclass.classes_[1]}: {svclass.n_support_[1]}")

        svs.append(svclass.n_support_)

    # Plot
    fig, ax = plt.subplots(1,1,figsize=(18,10))
    ax.plot(C, [i[0] for i in svs], color='orange', linewidth=2, label="class 0")
    ax.plot(C, [i[0] for i in svs], 'ro')
    ax.plot(C, [i[1] for i in svs], 'c-', linewidth=2, label="class 1")
    ax.plot(C, [i[1] for i in svs], 'bo')
    ax.set_title("C vs SV")
    ax.set_xlabel("Regularization Parameter (C)")
    ax.set_ylabel("Support Vectors")
    ax.legend()
    plt.show()

# GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
def q5d_svm_kernel_tuning():
    print("\n----- (d) SVM with Multiple Kernels -----")
    C = [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
    D = [1,2,3,4,5]
    C0 = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
    G = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3]
    param_grid = {"C": C, "kernel": ["linear"]}
    grid_ker = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', refit=True, verbose=3)
    grid_ker.fit(X_train, Y_train)
    param_grid = {"C": C, "kernel": ["poly"], "coef0":C0, "degree":D}
    grid_poly = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', refit=True, verbose=3)
    grid_poly.fit(X_train, Y_train)
    param_grid = {"C": C, "kernel": ["rbf"], "gamma":G}
    grid_rbf = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', refit=True, verbose=3)
    grid_rbf.fit(X_train, Y_train)
    param_grid = {"C": C, "kernel": ["sigmoid"], "coef0":C0, "gamma":G}
    grid_sigm = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', refit=True, verbose=3)
    grid_sigm.fit(X_train, Y_train)

    print("Kernel Best Param:", grid_ker.best_params_)
    best_ker = grid_ker.best_estimator_
    print("Poly Best Param:", grid_poly.best_params_)
    best_poly = grid_poly.best_estimator_
    print("RBF Best Param:", grid_rbf.best_params_)
    best_rbf = grid_rbf.best_estimator_
    print("Sigmoid Best Param:", grid_sigm.best_params_)
    best_sigm = grid_sigm.best_estimator_

    # Prediction
    kers = [best_ker, best_poly, best_rbf, best_sigm]
    all_metri = pd.DataFrame()
    for k in kers:
        Y_pred = k.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision = metrics.precision_score(Y_test, Y_pred)
        recall = metrics.recall_score(Y_test, Y_pred)
        f1_measure = metrics.f1_score(Y_test, Y_pred)
        data = {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1 Measure': f1_measure}
        metri = pd.DataFrame(data=data, index=[k.get_params()['kernel']])
        all_metri = pd.concat([all_metri, metri])

    print(all_metri)

# main function
if __name__=="__main__":
    df = q5a_load_db(filename)
    q5b_stratified_sampling(df)
    q5c_svm_linear_kernel()
    q5d_svm_kernel_tuning()

##### OUTPUT #####
"""
----- (a) Data Loading -----
Positive Samples= 125 
Negative Samples= 125

----- (b) Stratified Sampling -----
Training:: Positive Samples= 100 ; Negative Samples= 100
Testing :: Positive Samples= 25 ; Negative Samples= 25

----- (c) SVM with Linear Kernel -----
***** C=0.1 *****
Support vectors for class 0.0: 63
Support vectors for class 1.0: 65
***** C=0.2 *****
Support vectors for class 0.0: 55
Support vectors for class 1.0: 56
***** C=0.3 *****
Support vectors for class 0.0: 52
Support vectors for class 1.0: 52
***** C=0.5 *****
Support vectors for class 0.0: 47
Support vectors for class 1.0: 49
***** C=1 *****
Support vectors for class 0.0: 43
Support vectors for class 1.0: 45
***** C=2 *****
Support vectors for class 0.0: 41
Support vectors for class 1.0: 40
***** C=3 *****
Support vectors for class 0.0: 40
Support vectors for class 1.0: 41
***** C=5 *****
Support vectors for class 0.0: 40
Support vectors for class 1.0: 38
***** C=10 *****
Support vectors for class 0.0: 38
Support vectors for class 1.0: 37

----- (d) SVM with Multiple Kernels -----
Linear Best Param: {'C': 0.1, 'kernel': 'linear'}
Poly Best Param: {'C': 0.3, 'coef0': 2, 'degree': 2, 'kernel': 'poly'}
RBF Best Param: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Sigmoid Best Param: {'C': 1, 'coef0': 1, 'gamma': 0.2, 'kernel': 'sigmoid'}

         Accuracy  Recall  Precision  F1 Measure
linear       0.76    0.78   0.750000    0.764706
poly         0.75    0.76   0.745098    0.752475
rbf          0.76    0.78   0.750000    0.764706
sigmoid      0.74    0.76   0.730769    0.745098
"""