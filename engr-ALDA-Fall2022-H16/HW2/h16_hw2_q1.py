import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# (a) Data Loading
print("(a) ----- Loading Data -----\n")
pca_train = pd.read_csv("pca_train.csv") # reading the pca_train dataset
pca_test = pd.read_csv("pca_test.csv") # reading the pca_test dataset
print("The size of the testing dataset =", pca_test.size)
print("The size of the training dataset =", pca_train.size)
train_class = pca_train['Class'].value_counts()
print("Total size of the training set by class--\n0:", train_class[0], "\n1:", train_class[1], "\n2:", train_class[2])
test_class = pca_test['Class'].value_counts()
print("Total size of the testing set by class--\n0:", test_class[0], "\n1:", test_class[1], "\n2:", test_class[2])

# ---------------------------------------- #

# (b) Preprocessing Data-Normalization
print("\n\n(b) ----- Preprocessing Data-Normalization -----\n")
y_n_train = pca_train['Class']
x_n_train = pca_train.drop(columns={'Class'}) # x_n_train variable holds training dataset without class column
scaler_n = MinMaxScaler()
x_n_train_scaler = scaler_n.fit_transform(x_n_train) # x_n_train_scaler variable holds the mormalized training dataset
x_n_train_norm_df = pd.DataFrame(x_n_train_scaler, columns=x_n_train.columns)

y_n_test = pca_test['Class']
x_n_test = pca_test.drop(columns={'Class'}) # x_n_test variable holds training dataset without class column
x_n_test_scaler = scaler_n.transform(x_n_test) # x_n_test_scaler variable holds the mormalized testing dataset
x_n_test_norm_df = pd.DataFrame(x_n_test_scaler, columns=x_n_test.columns)

# i. Calculate the covariance matrix of the NEW training dataset.
x_n_train_cov = np.cov(np.transpose(x_n_train_norm_df)) # x_n_train_cov variable holds covariance of the normalized training dataset
  # 1) specify the dimension of the resulted covariance matrix
print('i.\nThe dimension of a covariance matrix =', np.shape(x_n_train_cov))
  # 2) given the space limitation, please report the first 5 ∗ 5 of the covariance matrix, that is, only reporting the first five rows and the first five columns of the entire covariance matrix.
print('\n5x5 Covariance matrix of training dataset--\n',x_n_train_cov[:5, :5])

# ii. Calculate the eigenvalues and the eigenvectors based on the entire covariance matrix in (i) above. Report the size of the covariance matrix and the 5 largest eigenvalues.
eigen_n_values_train, eigen_n_vector_train = eig(x_n_train_cov)
eigen_n_values_train = np.sort(eigen_n_values_train)[::-1]
eigen_n_vector_train = np.sort(eigen_n_vector_train)[::-1]
print('\nii.\nTop eigen values of the training dataset--\n', ", ".join(str(ev) for ev in eigen_n_values_train[:5]))

# iii. Display the eigenvalues using a bar graph or a plot, and choose a reasonable number(s) of eigenvectors. Justify your answer.
x_n = np.arange(len(x_n_train_cov))+1
f1, ax1 = plt.subplots(1,1)
ax1.bar(x_n, height=eigen_n_values_train, width=0.2, color='green', alpha=0.4)
ax1.plot(x_n, eigen_n_values_train, 'r-', linewidth=2, alpha=0.5)
ax1.plot(x_n, eigen_n_values_train, 'bo', markersize=2, alpha=1)
ax1.set_title('Scree Plot for eigenvalues of Normalized data')
ax1.set_xlabel('Principal Components', fontsize=8)
ax1.set_ylabel('Eigenvalue', fontsize=8)

eigen_n_values_train_pct = eigen_n_values_train / np.sum(eigen_n_values_train) * 100
print("\niii.\nPercentage Matrix--", eigen_n_values_train_pct)
print("\nHence we will select '", sum(et_s > 10 for et_s in eigen_n_values_train_pct), "' eigen values as those are greater than 10% of the variance i.e. \"PC", " ; \"PC ".join((str(i) + "\" (" + str(ep)+"%)") for i, ep in enumerate(eigen_n_values_train_pct) if ep > 10))

# iv. PCA & KNN
principal_components = [1, 2, 3, 5, 10, 13]
accuracy_n = {}
accuracy_n_list=[]
y_n_pred_5 = []
for k in principal_components:
  pca_n = PCA(n_components=k)
  classifier_n = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
  classifier_n.fit(pca_n.fit_transform(x_n_train_norm_df), y_n_train)
  y_n_pred = classifier_n.predict(pca_n.transform(x_n_test_norm_df))
  accuracy_n[k] = metrics.accuracy_score(y_n_test, y_n_pred)
  accuracy_n_list.append(metrics.accuracy_score(y_n_test, y_n_pred))
  if k==5:
    y_n_pred_5 = y_n_pred
    y_n_pca_test_5 = pca_n.transform(x_n_test_norm_df)
print('\niv.\nAll Accuracies for Normalized Data:', accuracy_n)
print('\nAccuracy of the classifier when principal components is 5 and number of neighbors is 5 =', accuracy_n[5])

automated_n_df = pd.DataFrame(y_n_pca_test_5, columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
automated_n_df = automated_n_df.join(y_n_test)
automated_n_df["Predicted Class"] = y_n_pred_5
automated_n_df.to_csv('pca_test_n_with_pred.csv')

f_nn, ax_nn = plt.subplots(1,1)
ax_nn.bar(principal_components, height=accuracy_n_list, color='green', alpha=0.4)
ax_nn.plot(principal_components, accuracy_n_list, color='black')
ax_nn.set_title('Accuracy of KNN Classifier on Normalized Data')
ax_nn.set_xlabel('Number of Components', fontsize=8)
ax_nn.set_ylabel('Accuracy of Prediction Classifier', fontsize=8)

# # ---------------------------------------- #

# (c) Preprocess Data-Standardization
print("\n\n(c) ----- Preprocess Data-Standardization -----\n")
y_s_train = pca_train['Class']
x_s_train = pca_train.drop(columns={'Class'}) # x_s_train variable holds training dataset without class column
scaler_s = StandardScaler()
x_s_train_scaler = scaler_s.fit_transform(x_s_train) # x_s_train_scaler variable holds the standardized training dataset
x_s_train_stdr_df = pd.DataFrame(x_s_train_scaler, columns=x_s_train.columns)

y_s_test = pca_test['Class']
x_s_test = pca_test.drop(columns={'Class'}) # x_s_test variable holds training dataset without class column
x_s_test_scaler = scaler_s.transform(x_s_test) # x_s_test_scaler variable holds the standardized testing dataset
x_s_test_stdr_df = pd.DataFrame(x_s_test_scaler, columns=x_s_test.columns)

# i. Calculate the covariance matrix of the NEW training dataset.
x_s_train_cov = np.cov(np.transpose(x_s_train_stdr_df)) # x_s_train_cov variable holds covariance of the standardized training dataset
  # 1) specify the dimension of the resulted covariance matrix
print('i.\nThe dimension of a covariance matrix =', np.shape(x_s_train_cov))
  # 2) given the space limitation, please report the first 5 ∗ 5 of the covariance matrix, that is, only reporting the first five rows and the first five columns of the entire covariance matrix.
print('\n5x5 Covariance matrix of training dataset--\n', x_s_train_cov[:5, :5])

# ii. Calculate the eigenvalues and the eigenvectors based on the entire covariance matrix in (i) above. Report the size of the covariance matrix and the 5 largest eigenvalues.
eigen_s_values_train, eigen_s_vector_train = eig(x_s_train_cov)
eigen_s_values_train = np.sort(eigen_s_values_train)[::-1]
eigen_s_vector_train = np.sort(eigen_s_vector_train)[::-1]
print('\nii.\nTop eigen values of the training dataset--\n', ", ".join(str(ev) for ev in eigen_s_values_train[:5]))

# iii. Display the eigenvalues using a bar graph or a plot, and choose a reasonable number(s) of eigenvectors. Justify your answer.
x_s = np.arange(len(x_s_train_cov))+1
f2, ax2 = plt.subplots(1,1)
ax2.bar(x_s, height=eigen_s_values_train, width=0.2, color='green', alpha=0.4)
ax2.plot(x_s, eigen_s_values_train, 'r-', linewidth=2, alpha=0.5)
ax2.plot(x_s, eigen_s_values_train, 'bo', markersize=2, alpha=1)
ax2.set_title('Scree Plot for eigenvalues of Standardized data')
ax2.set_xlabel('Principal Components', fontsize=8)
ax2.set_ylabel('Eigenvalue', fontsize=8)

eigen_s_values_train_pct = eigen_s_values_train / np.sum(eigen_s_values_train) * 100
print("\niii.\nPercentage Matrix--", eigen_s_values_train_pct)
print("\nHence we will select '", sum(et_s > 10 for et_s in eigen_s_values_train_pct), "' eigen values as those are greater than 10% of the variance i.e. \"PC", "; \"PC ".join((str(i) + "\"(" + str(ep)+"%)") for i, ep in enumerate(eigen_s_values_train_pct) if ep > 10))

# iv.
# • Next, you will combine PCA with a K-nearest neighbor (KNN) classifier. More specifically, PCA will be applied to reduce the dimensionality of data by transforming the original data into p (p ≤ 13) principal components; and then KNN (K = 5, euclidean distance as distance metric) will be employed to the p principal components for classification.
# • Report the accuracy of the NEW testing dataset when using PCA (p = 5) with 5NN. To show your work, please submit the corresponding .csv file (including the name of .csv file in your answer below). Your .csv file should have 7 columns: columns 1-5 are the 5 principal components, column 6 is the original ground truth output “Class”, and the last column is the predicted output “Class”.
# • Plot your results by varying p: 1, 2, 3, 5, 10, and 13 respectively. In your plot, the x-axis represents the number of principal components and the y-axis refers to the accuracy of the NEW testing dataset using the corresponding number of principal components and 5NN.
# • Based upon the (PCA + 5NN)’s results above, what is the most “reasonable” number of principal components among all the choices? Justify your answer.
principal_components = [1, 2, 3, 5, 10, 13]
accuracy_s = {}
accuracy_s_list=[]
y_s_pred_5 = []
for k in principal_components:
  pca_s = PCA(n_components=k)
  classifier_s = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
  classifier_s.fit(pca_s.fit_transform(x_s_train_stdr_df), y_s_train)
  y_s_pred = classifier_s.predict(pca_s.transform(x_s_test_stdr_df))
  accuracy_s[k] = metrics.accuracy_score(y_s_test, y_s_pred)
  accuracy_s_list.append(metrics.accuracy_score(y_s_test, y_s_pred))
  if k==5:
    y_s_pred_5 = y_s_pred
    y_s_pca_test_5 = pca_s.transform(x_s_test_stdr_df)
print('\niv.\nAll Accuracies for Standardized Data:', accuracy_s)
print('\nAccuracy of the classifier when principal components is 5 and number of neighbors is 5 =', accuracy_s[5])

automated_s_df = pd.DataFrame(y_s_pca_test_5, columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
automated_s_df = automated_s_df.join(y_s_test)
automated_s_df["Predicted Class"] = y_s_pred_5
automated_s_df.to_csv('pca_test_s_with_pred.csv')

f_ss, ax_ss = plt.subplots(1,1)
ax_ss.bar(principal_components, height=accuracy_s_list, color='red', alpha=0.4)
ax_ss.plot(principal_components, accuracy_s_list, color='blue')
ax_ss.set_title('Accuracy of KNN Classifier on Standardized Data')
ax_ss.set_xlabel('Number of Components', fontsize=8)
ax_ss.set_ylabel('Accuracy of Prediction Classifier', fontsize=8)
plt.show()

##### OUTPUT #####
"""
(a) ----- Loading Data -----

The size of the testing dataset = 742
The size of the training dataset = 1750
Total size of the training set by class--
0: 37 
1: 51 
2: 37
Total size of the testing set by class--
0: 22 
1: 20 
2: 11


(b) ----- Preprocessing Data-Normalization -----

i.
The dimension of a covariance matrix = (13, 13)

5x5 Covariance matrix of training dataset--
 [[ 0.05437915  0.00536061  0.00578453 -0.01417342  0.01611634]
 [ 0.00536061  0.05209397  0.00683929  0.01195127 -0.00075704]
 [ 0.00578453  0.00683929  0.02204741  0.01117852  0.0110121 ]
 [-0.01417342  0.01195127  0.01117852  0.02998937 -0.00311085]
 [ 0.01611634 -0.00075704  0.0110121  -0.00311085  0.03856216]]

ii.
Top eigen values of the training dataset--
 0.23832336476046134, 0.1262850346656315, 0.05035845207718303, 0.04003988048566648, 0.03664120361873718

iii.
Percentage Matrix-- [39.81856132 21.09943522  8.41378316  6.68977814  6.12193443  4.46693768
  3.51181202  2.48067784  2.13608214  1.93173295  1.4142445   1.09643265
  0.81858796]

Hence we will select ' 2 ' eigen values as those are greater than 10% of the variance i.e. "PC 0" (39.8185613214266%) ; "PC 1" (21.09943522266926%)

iv.
All Accuracies for Normalized Data: {1: 0.7547169811320755, 2: 0.9811320754716981, 3: 0.9622641509433962, 5: 0.9622641509433962, 10: 0.9811320754716981, 13: 0.9622641509433962}

Accuracy of the classifier when principal components is 5 and number of neighbors is 5 = 0.9622641509433962


(c) ----- Preprocess Data-Standardization -----

i.
The dimension of a covariance matrix = (13, 13)

5x5 Covariance matrix of training dataset--
 [[ 1.00806452  0.10152955  0.16840753 -0.35380446  0.35477897]
 [ 0.10152955  1.00806452  0.20343545  0.30480694 -0.01702669]
 [ 0.16840753  0.20343545  1.00806452  0.43823826  0.38071409]
 [-0.35380446  0.30480694  0.43823826  1.00806452 -0.09221525]
 [ 0.35477897 -0.01702669  0.38071409 -0.09221525  1.00806452]]

ii.
Top eigen values of the training dataset--
 4.759134655006301, 2.6488837008249813, 1.4796208845395886, 1.0129897334928881, 0.7721873479912549

iii.
Percentage Matrix-- [36.31585829 20.21302024 11.29064552  7.72989089  5.89238346  4.26224239
  3.91986034  2.5504865   2.30565529  1.98414205  1.66114905  1.06844201
  0.80622396]

Hence we will select ' 3 ' eigen values as those are greater than 10% of the variance i.e. "PC 0"(36.31585829050962%); "PC 1"(20.21302024014139%); "PC 2"(11.290645518948244%)

iv.
All Accuracies for Standardized Data: {1: 0.7735849056603774, 2: 0.9811320754716981, 3: 0.9622641509433962, 5: 0.9245283018867925, 10: 0.9433962264150944, 13: 0.9811320754716981}

Accuracy of the classifier when principal components is 5 and number of neighbors is 5 = 0.9245283018867925
"""