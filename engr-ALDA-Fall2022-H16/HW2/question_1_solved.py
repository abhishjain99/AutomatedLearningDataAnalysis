#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Reading the data
train_data_df = pd.read_csv("pca_train.csv")
test_data_df = pd.read_csv("pca_test.csv")

#Output format
def print_shape_and_class_occurences(df):
    print ("(Rows, Columns) : {}".format(df.shape))
    class_counts = df['Class'].value_counts()
    print("Count of Class 0: ",class_counts[0])
    print("Count of Class 1: ", class_counts[1])
    print("Count of Class 2: ", class_counts[2])
    
print("\nTraining data information:")
print_shape_and_class_occurences(train_data_df)

print("\nTesting data information:")
print_shape_and_class_occurences(test_data_df)

#Normalization
scaler = MinMaxScaler()
norm_train_data_df = pd.DataFrame(scaler.fit_transform(train_data_df.loc[:,train_data_df.columns != 'Class']))
norm_test_data_df = pd.DataFrame(scaler.transform(test_data_df.loc[:,test_data_df.columns != 'Class']))

#Covariance matrix
norm_train_cov = norm_train_data_df.cov()
print ("\nDimensions of Covariance matrix of normalized training data (Rows, Columns) : {}".format(norm_train_cov.shape))
print (norm_train_cov.iloc[0:5,0:5])

#Eigen values and vectors
eig_val,eig_vec = np.linalg.eig(norm_train_cov)
pd.DataFrame(eig_val,columns=['Value']).sort_values(by=['Value'],ascending=False)
print("\nLargest 5 Eigenvalues:\n{}".format(pd.DataFrame(eig_val,columns=['Value']).sort_values(by=['Value'],ascending=False).loc[0:4,'Value'].to_string(index=False)))
fig = plt.figure(figsize = (10, 5))
fig.suptitle('Eigen values for Normalized data')
plt.plot(eig_val, '.r-') 
plt.xlabel("Eigen Vectors")
plt.ylabel("Eigen Values")

#Selecting significant Eigen values
eig_val_percentage = (eig_val / eig_val.sum())*100
print("\nPercentage of variance contributed by each Eigen vector:\n {}".format(eig_val_percentage))
print("\nSince only two eigen values are contributing more than 10% of the variance, we will be selecting the first 2 eigen values only.")

#PCA transformation
pca_1 = PCA(n_components = 1)
pca_2 = PCA(n_components = 2)
pca_3 = PCA(n_components = 3)
pca_5 = PCA(n_components = 5)
pca_10 = PCA(n_components = 10)
pca_13 = PCA(n_components = 13)


def pca_transform(pca,train,test):
    norm_pca_train = pca.fit_transform(train)
    norm_pca_test = pca.transform(test)
    return norm_pca_train,norm_pca_test

def knn_train_and_predict(knn,train_X,train_Y,test_X):
    knn.fit(train_X,train_Y)
    return knn.predict(test_X)
    
    
norm_pca_1_train , norm_pca_1_test = pca_transform(pca_1,norm_train_data_df,norm_test_data_df)
norm_pca_2_train , norm_pca_2_test = pca_transform(pca_2,norm_train_data_df,norm_test_data_df)
norm_pca_3_train , norm_pca_3_test = pca_transform(pca_3,norm_train_data_df,norm_test_data_df)
norm_pca_5_train , norm_pca_5_test = pca_transform(pca_5,norm_train_data_df,norm_test_data_df)
norm_pca_10_train , norm_pca_10_test = pca_transform(pca_10,norm_train_data_df,norm_test_data_df)
norm_pca_13_train , norm_pca_13_test = pca_transform(pca_13,norm_train_data_df,norm_test_data_df)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

train_data_output = train_data_df['Class']
test_data_output = test_data_df['Class'] 

test_data_predict_output_knn1 = knn_train_and_predict(knn, norm_pca_1_train, train_data_output, norm_pca_1_test) 
test_data_predict_output_knn2 = knn_train_and_predict(knn, norm_pca_2_train, train_data_output, norm_pca_2_test)
test_data_predict_output_knn3 = knn_train_and_predict(knn, norm_pca_3_train, train_data_output, norm_pca_3_test)
test_data_predict_output_knn5 = knn_train_and_predict(knn, norm_pca_5_train, train_data_output, norm_pca_5_test) 
test_data_predict_output_knn10 = knn_train_and_predict(knn, norm_pca_10_train, train_data_output, norm_pca_10_test)
test_data_predict_output_knn13 = knn_train_and_predict(knn, norm_pca_13_train, train_data_output, norm_pca_13_test)

#Accuracy calculation
accuracy = {}
accuracy["PC 1"] = accuracy_score(test_data_predict_output_knn1, test_data_output)
accuracy["PC 2"] = accuracy_score(test_data_predict_output_knn2, test_data_output)
accuracy["PC 3"] = accuracy_score(test_data_predict_output_knn3, test_data_output)
accuracy["PC 5"] = accuracy_score(test_data_predict_output_knn5, test_data_output)
accuracy["PC 10"] = accuracy_score(test_data_predict_output_knn10, test_data_output)
accuracy["PC 13"] = accuracy_score(test_data_predict_output_knn13, test_data_output)  
print("\nAccuracy on Normalized data: \n {}".format(accuracy))

knn_5_accuracy_df = pd.DataFrame(norm_pca_5_test,columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5']).join(test_data_output).rename({'Class':'Ground Truth'},axis=1)
knn_5_accuracy_df['Predicted Output'] = test_data_predict_output_knn5
knn_5_accuracy_df.to_csv('5NN Accuracy for Normalized data.csv',index=False)
print ('\nSample 5 rows of the 5NN Accuracy.csv file:\n{}'.format(knn_5_accuracy_df.iloc[0:5,:]))

fig = plt.figure(figsize = (10, 5))
fig.suptitle('Accuracy of KNN on Normalized data')
plt.bar(range(len(accuracy)),list(accuracy.values()),align='center', color ='maroon',width = 0.4)
plt.xticks(range(len(accuracy)),list(accuracy.keys()))
plt.xlabel('KNN on different number of principal components')
plt.ylabel('Accuracy')


#Standardization
scaler = StandardScaler()
std_train_data_df = pd.DataFrame(scaler.fit_transform(train_data_df.loc[:,train_data_df.columns != 'Class']))
std_test_data_df = pd.DataFrame(scaler.transform(test_data_df.loc[:,test_data_df.columns != 'Class']))

#Covariance matrix
std_train_cov = std_train_data_df.cov()
print ("\nDimensions of Covariance matrix of standardized training data (Rows, Columns) : {}".format(norm_train_cov.shape))
print (std_train_cov.iloc[0:5,0:5])

#Eigen values and vectors
eig_val,eig_vec = np.linalg.eig(std_train_cov)
pd.DataFrame(eig_val,columns=['Value']).sort_values(by=['Value'],ascending=False)
print("\nLargest 5 Eigenvalues:\n{}".format(pd.DataFrame(eig_val,columns=['Value']).sort_values(by=['Value'],ascending=False).loc[0:4,'Value'].to_string(index=False)))
fig = plt.figure(figsize = (10, 5))
fig.suptitle('Eigen Values for Standardized data')
plt.plot(eig_val, '.r-')
plt.xlabel("Eigen Vectors")
plt.ylabel("Eigen Values")

#Selecting significant Eigen values
eig_val_percentage = (eig_val / eig_val.sum())*100
print("\nPercentage of variance contributed by each Eigen vector:\n {}".format(eig_val_percentage))
print("\nSince only three eigen values are contributing more than 10% of the variance, we will be selecting the first 3 eigen values only.")

std_pca_1_train , std_pca_1_test = pca_transform(pca_1,std_train_data_df,std_test_data_df)
std_pca_2_train , std_pca_2_test = pca_transform(pca_2,std_train_data_df,std_test_data_df)
std_pca_3_train , std_pca_3_test = pca_transform(pca_3,std_train_data_df,std_test_data_df)
std_pca_5_train , std_pca_5_test = pca_transform(pca_5,std_train_data_df,std_test_data_df)
std_pca_10_train , std_pca_10_test = pca_transform(pca_10,std_train_data_df,std_test_data_df)
std_pca_13_train , std_pca_13_test = pca_transform(pca_13,std_train_data_df,std_test_data_df)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

train_data_output = train_data_df['Class']
test_data_output = test_data_df['Class'] 

test_data_predict_output_knn1 = knn_train_and_predict(knn, std_pca_1_train, train_data_output, std_pca_1_test) 
test_data_predict_output_knn2 = knn_train_and_predict(knn, std_pca_2_train, train_data_output, std_pca_2_test)
test_data_predict_output_knn3 = knn_train_and_predict(knn, std_pca_3_train, train_data_output, std_pca_3_test)
test_data_predict_output_knn5 = knn_train_and_predict(knn, std_pca_5_train, train_data_output, std_pca_5_test) 
test_data_predict_output_knn10 = knn_train_and_predict(knn, std_pca_10_train, train_data_output, std_pca_10_test)
test_data_predict_output_knn13 = knn_train_and_predict(knn, std_pca_13_train, train_data_output, std_pca_13_test)

#Accuracy calculation
accuracy = {}
accuracy["PC 1"] = accuracy_score(test_data_predict_output_knn1, test_data_output)
accuracy["PC 2"] = accuracy_score(test_data_predict_output_knn2, test_data_output)
accuracy["PC 3"] = accuracy_score(test_data_predict_output_knn3, test_data_output)
accuracy["PC 5"] = accuracy_score(test_data_predict_output_knn5, test_data_output)
accuracy["PC 10"] = accuracy_score(test_data_predict_output_knn10, test_data_output)
accuracy["PC 13"] = accuracy_score(test_data_predict_output_knn13, test_data_output)
print("\nAccuracy on Standardized data: \n {}".format(accuracy))    

knn_5_accuracy_df = pd.DataFrame(std_pca_5_test,columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5']).join(test_data_output).rename({'Class':'Ground Truth'},axis=1)
knn_5_accuracy_df['Predicted Output'] = test_data_predict_output_knn5
knn_5_accuracy_df.to_csv('5NN Accuracy for Standardized data.csv',index=False)
print ('\nSample 5 rows of the 5NN Accuracy.csv file:\n{}'.format(knn_5_accuracy_df.iloc[0:5,:]))

fig = plt.figure(figsize = (10, 5))
fig.suptitle('Accuracy of KNN on Standardized data')
plt.bar(range(len(accuracy)),list(accuracy.values()),align='center', color ='maroon',width = 0.4)
plt.xticks(range(len(accuracy)),list(accuracy.keys()))
plt.xlabel('KNN on different number of principal components')
plt.ylabel('Accuracy')

plt.show()