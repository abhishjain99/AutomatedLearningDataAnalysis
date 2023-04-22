# pandas Documentation - https://pandas.pydata.org/docs/
# numpy Reference Documentation - https://numpy.org/doc/stable/reference/index.html
# matplotlib Reference Documentation - https://matplotlib.org/2.0.2/index.html
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# (a) What is the leave-one-out cross-validation error of 1NN on this dataset?
print("(a) ----- Leave-One-Out Cross-Validation Error of 1NN -----\n")
og_data_df = pd.read_csv("h3_q1.csv")
error_list_1n = []
for t in range(len(og_data_df)):
  # Creating Training and Testing Datasets
  # * Removing record t from dataset *
  tr_df_1n_1n = pd.DataFrame().assign(x1=og_data_df['x1'][0:t], x2=og_data_df['x2'][0:t], y=og_data_df['y'][0:t])
  tr_df2_1n = pd.DataFrame().assign(x1=og_data_df['x1'][t+1:], x2=og_data_df['x2'][t+1:], y=og_data_df['y'][t+1:])
  training_data_df_1n = pd.concat([tr_df_1n_1n, tr_df2_1n])
  testing_data_df_1n = pd.DataFrame([[og_data_df['x1'][t], og_data_df['x2'][t], og_data_df['y'][t]]], columns=['x1', 'x2', 'y'])

  x_train_1n = training_data_df_1n[['x1', 'x2']]
  y_train_1n = training_data_df_1n['y']
  x_test_1n = testing_data_df_1n[['x1', 'x2']]
  y_test_1n = testing_data_df_1n['y']
  
  if t==2:
    id3_df_3n = x_train_1n
  if t==4:
    id5_df_3n = x_train_1n

  # Using KNN with Euclidean distance and calculating error
  # * Training on remaining dataset after removing record t *
  classifier_1n = KNeighborsClassifier(n_neighbors=1, metric='euclidean') # default p=2 / metric='euclidean'
  classifier_1n.fit(x_train_1n, y_train_1n)
  y_pred_1n = classifier_1n.predict(x_test_1n)
  # * Calculating error on record t *
  y_error_1n = abs(y_test_1n - y_pred_1n).to_list()[0]
  error_list_1n.append(y_error_1n)
  print('ID:', t+1, '- predicted_y:', y_pred_1n[0], ', actual_y:', y_test_1n.to_list()[0], ', error_y:', y_error_1n)

  # Plotting dataset with prediction
  f_1n, ax_1n = plt.subplots(1,1)
  ax_1n.scatter(x_train_1n['x1'], x_train_1n['x2'], color='g', alpha=0.4)
  ax_1n.scatter(x_test_1n['x1'], x_test_1n['x2'], c=y_pred_1n, s=50, alpha=1)
  ax_1n.set_title('LOOCV with 1NN : ID='+str(t+1), fontsize=10)
  ax_1n.set_xlabel('x1', fontsize=8)
  ax_1n.set_ylabel('x2', fontsize=8)

# * Done with all points. So reporting the final mean error *
print('\nLeave-One-Out Cross-Validation Error of 1NN =', np.mean(error_list_1n))

# ---------------------------------------- #

# (b) What are the 3 nearest neighbors for data points 3 and 5 respectively.
print("\n\n(b) ----- 3 Nearest Neighbors -----\n")
n = [3, 5]
for k in n:
  # Calculating and sorting Euclidean distance
  close_eucledian_points = []
  tuple_data_3n = [(a, b) for a, b in id3_df_3n.values] if k == 3 else [(a, b) for a, b in id5_df_3n.values]
  P = [og_data_df.iloc[k-1]['x1'], og_data_df.iloc[k-1]['x2']]
  euc_dist = [np.sqrt((P[0] - a[0]) ** 2 + (P[1] - a[1]) ** 2) for a in tuple_data_3n]
  close_eucledian_points = [tuple_data_3n[i] for i in np.argsort(euc_dist)]
  print('3 Nearest Neighbors for ID =', k, '---')
  for e in close_eucledian_points[0:3]:
    coord_str_3n = 'x1 ==' + str(e[0]) + '& x2 ==' + str(e[1])
    id_3n = og_data_df[og_data_df.eval(coord_str_3n)].index.to_list()[0]+1
    print("ID", id_3n, ":", e)

  # Plotting 3 nearest neighbors
  f3, ax3 = plt.subplots(1,1)
  for x, y in close_eucledian_points:
      ax3.scatter(x, y, color='green', marker='*', s=20, alpha=0.3)
  for x, y in close_eucledian_points[0:3]:
      ax3.scatter(x, y, color='black', marker='o', s=30, alpha=0.7)
  ax3.scatter(P[0], P[1], color='magenta', marker='D', s=40, alpha = 0.6)
  ax3.set_xlabel('x1')
  ax3.set_ylabel('x2')
  ax3.set_title('Nearest 3 Neighbors for ID='+str(k))

  plt.show()
# # ---------------------------------------- #

# (c) What is the 3-folded cross-validation error of 3NN on this dataset? For the ith fold, the testing dataset is composed of all the data points whose (ID mod 3 = i − 1).
print("\n\n(c) ----- 3-folded Cross-Validation Error of 3NN -----\n")
main_df = pd.read_csv("h3_q1.csv")

# 3 fold
df1 = main_df[main_df['ID']%3 == 0]
df2 = main_df[main_df['ID']%3 == 1]
df3 = main_df[main_df['ID']%3 == 2]

# Case - 1 when i = 0 => df1 is test, df2 and df3  are train datasets
df1_x_test = df1[['x1','x2']]
df1_y_test = df1['y']
df2_df3_merge = pd.concat([df2,df3])
df1_x_train = df2_df3_merge[['x1','x2']]
df1_y_train = df2_df3_merge['y']

classifier_3n_1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # default p=2 / metric='euclidean'
classifier_3n_1.fit(df1_x_train, df1_y_train)
y_pred_3n_1 = classifier_3n_1.predict(df1_x_test)
# * Calculating error on record t *
y_error_3n_1 = np.square(np.subtract(df1_y_test,y_pred_3n_1)).mean()
error_list_1n.append(y_error_3n_1)
print('Fold 1 when i = 0 ','- predicted_y_1:', y_pred_3n_1, ', actual_y_1:', df1_y_test.to_list(), ', error_y_1:', y_error_3n_1)

# Plotting dataset with prediction for Fold 1 when i = 0
f_fold1n, ax_fold1n = plt.subplots(1,1)
ax_fold1n.scatter(df1_x_train['x1'], df1_x_train['x2'], color='g', alpha=0.4)
ax_fold1n.scatter(df1_x_test['x1'], df1_x_test['x2'], c=y_pred_3n_1, s=50, alpha=1)
ax_fold1n.set_title('CV of 3NN when i=0 and ID = 3,6,9', fontsize=10)
ax_fold1n.set_xlabecl('x1', fontsize=8)
ax_fold1n.set_ylabel('x2', fontsize=8)
plt.show()

# Case - 2 when i = 0 => df2 is test, df1 and df3  are train datasets
df2_x_test = df2[['x1','x2']]
df2_y_test = df2['y']
df1_df3_merge = pd.concat([df1,df3])
df2_x_train = df1_df3_merge[['x1','x2']]
df2_y_train = df1_df3_merge['y']

classifier_3n_2 = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # default p=2 / metric='euclidean'
classifier_3n_2.fit(df2_x_train, df2_y_train)
y_pred_3n_2 = classifier_3n_2.predict(df2_x_test)
# * Calculating error on record t *
y_error_3n_2 = np.square(np.subtract(df2_y_test,y_pred_3n_2)).mean()
error_list_1n.append(y_error_3n_2)
print('Fold 2 when i = 1 ','- predicted_y_2:', y_pred_3n_2, ', actual_y_2:', df2_y_test.to_list(), ', error_y_2:', y_error_3n_2)

# Plotting dataset with prediction for Fold 1 when i = 1
f_fold2n, ax_fold2n = plt.subplots(1,1)
ax_fold2n.scatter(df2_x_train['x1'], df2_x_train['x2'], color='g', alpha=0.4)
ax_fold2n.scatter(df2_x_test['x1'], df2_x_test['x2'], c=y_pred_3n_2, s=50, alpha=1)
ax_fold2n.set_title('CV of 3NN when i=1 and ID = 1,4,7,10', fontsize=10)
ax_fold2n.set_xlabel('x1', fontsize=8)
ax_fold2n.set_ylabel('x2', fontsize=8)
plt.show()

# Case - 3 when i = 0 => df3 is test, df1 and df2  are train datasets
df3_x_test = df3[['x1','x2']]
df3_y_test = df3['y']
df1_df2_merge = pd.concat([df1,df2])
df3_x_train = df1_df2_merge[['x1','x2']]
df3_y_train = df1_df2_merge['y']

classifier_3n_3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # default p=2 / metric='euclidean'
classifier_3n_3.fit(df3_x_train, df3_y_train)
y_pred_3n_3 = classifier_3n_3.predict(df3_x_test)
# * Calculating error on record t *
y_error_3n_3 = np.square(np.subtract(df3_y_test,y_pred_3n_3)).mean()
error_list_1n.append(y_error_3n_3)
print('Fold 3 when i = 2 ','- predicted_y_3:', y_pred_3n_3, ', actual_y_3:', df3_y_test.to_list(), ', error_y_3:', y_error_3n_3)

# Plotting dataset with prediction for Fold 1 when i = 2
f_fold3n, ax_fold3n = plt.subplots(1,1)
ax_fold3n.scatter(df3_x_train['x1'], df3_x_train['x2'], color='g', alpha=0.4)
ax_fold3n.scatter(df3_x_test['x1'], df3_x_test['x2'], c=y_pred_3n_3, s=50, alpha=1)
ax_fold3n.set_title('CV of 3NN when i=1 and ID = 2,5,8', fontsize=10)
ax_fold3n.set_xlabel('x1', fontsize=8)
ax_fold3n.set_ylabel('x2', fontsize=8)
plt.show()

print("Final Error in 3NN :", (y_error_3n_1+y_error_3n_2+y_error_3n_3)/3)
##### OUTPUT #####
"""
(a) ----- Leave-One-Out Cross-Validation Error of 1NN -----

ID: 1 - predicted_y: 0 , actual_y: 0 , error_y: 0
ID: 2 - predicted_y: 1 , actual_y: 1 , error_y: 0
ID: 3 - predicted_y: 1 , actual_y: 1 , error_y: 0
ID: 4 - predicted_y: 1 , actual_y: 1 , error_y: 0
ID: 5 - predicted_y: 0 , actual_y: 0 , error_y: 0
ID: 6 - predicted_y: 1 , actual_y: 1 , error_y: 0
ID: 7 - predicted_y: 1 , actual_y: 0 , error_y: 1
ID: 8 - predicted_y: 1 , actual_y: 1 , error_y: 0
ID: 9 - predicted_y: 0 , actual_y: 0 , error_y: 0
ID: 10 - predicted_y: 0 , actual_y: 0 , error_y: 0

Leave-One-Out Cross-Validation Error of 1NN = 0.1


(b) ----- 3 Nearest Neighbors -----

3 Nearest Neighbors for ID = 3 ---
ID 8 : (5.49, 3.0)
ID 2 : (6.9, 5.0)
ID 5 : (8.16, 0.0)
3 Nearest Neighbors for ID = 5 ---
ID 1 : (8.18, 0.0)
ID 8 : (5.49, 3.0)
ID 3 : (5.67, 4.0)


(c) ----- 3-folded Cross-Validation Error of 3NN -----
Fold 1 when i = 0  - predicted_y_1: [1 1 0] , actual_y_1: [1, 1, 0] , error_y_1: 0.0
Fold 2 when i = 1  - predicted_y_2: [1 1 1 0] , actual_y_2: [0, 1, 0, 0] , error_y_2: 0.5
Fold 3 when i = 2  - predicted_y_3: [0 0 0] , actual_y_3: [1, 0, 1] , error_y_3: 0.6666666666666666
Final Error in 3NN : 0.38888888888888884
"""