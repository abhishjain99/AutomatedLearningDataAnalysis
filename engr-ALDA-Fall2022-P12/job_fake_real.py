# Dataset - https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
# pandas Documentation - https://pandas.pydata.org/docs/
# numpy Reference Documentation - https://numpy.org/doc/stable/reference/index.html
# matplotlib Reference Documentation - https://matplotlib.org/2.0.2/index.html
# seaborn - https://seaborn.pydata.org/index.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


import warnings
warnings.filterwarnings("ignore")

### DATA SELECTION/READ ###
filename = "P12_fake_job_postings.csv"
def read_data(filename):
    dataset = pd.read_csv(filename)
    dataset_no_id = dataset.drop(columns=['job_id'])
    print(dataset_no_id.info())
    return dataset_no_id

### DATA VISUALIZATION ###
def intial_data_visualization(dataset):
    # Getting unique values count
    dataset_no_label = dataset.copy().drop(columns=['fraudulent'])
    unique_vals = [dataset_no_label[col].unique().size for col in dataset_no_label.columns]

    f_unique, ax_unique = plt.subplots(1,1)
    ax_unique.plot(list(dataset_no_label.columns), unique_vals, "k-")
    ax_unique.plot(list(dataset_no_label.columns), unique_vals, "ro")
    ax_unique.set_title("Unique values - attributewise")
    ax_unique.set_xlabel("Attributes")
    ax_unique.set_ylabel("Unique Counts")
    ax_unique.set_xticklabels(list(dataset_no_label.columns), rotation=30)

    real_vals = (dataset["fraudulent"]==0).sum()
    fake_vals = (dataset["fraudulent"]==1).sum()
    labels = ["Real", "Fake"]
    vals = [real_vals, fake_vals]
    bar_colors=["#374EA2", "#EF4129"]
    f_fraud, ax_fraud = plt.subplots(1,1)
    ax_fraud.bar(labels, vals, label=labels, color=bar_colors)
    ax_fraud.set_ylabel("Count")
    ax_fraud.legend(title="Job Postings")


### DATA PREPROCESSING ###
def preprocess_data(dataset):
    # fill null, not applicable and unspecified values with 'No Info'
    print(dataset.isna().sum().sum())
    dataset.fillna('No Info', inplace=True)
    print(dataset.info())
    dataset = dataset.replace(['Not Applicable', 'Unspecified'], 'No Info')

    # removing duplicate entries
    dup_df = dataset[dataset.duplicated()]
    no_dup_df = dataset.drop_duplicates()
    print("duplicated rows: ", list(dup_df.index))
    print("Duplicated Rows Count=", len(list(dup_df.index)))
    print(no_dup_df.info())

    # only keeping selected columns
    df_selected_cols = no_dup_df[['title', 'location', 'department', 'salary_range' , 'company_profile', 'employment_type', 'required_experience', 'required_education', 'industry', 'function', 'fraudulent']]
    df_ccount = len(df_selected_cols)
    print('Count after duplicates removed=', df_ccount) #17598

    # categorical to numerical conversion - https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
    category_cols = df_selected_cols.select_dtypes(['object']).columns
    df_selected_cols[category_cols] = df_selected_cols[category_cols].apply(lambda x: pd.factorize(x)[0]) # codes, uniques = pd.factorize(cat)
    print(df_selected_cols)
    df_selected_cols.to_csv("ds.csv")
    dfs = { "no_dup_df": no_dup_df, "selected_columns_dataset": df_selected_cols }
    return dfs


def heatmap_correlation(df):
    # using heatmap to see correlation between data points
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    f_heatmap, ax_heatmap = plt.subplots(1,1)
    i = sns.heatmap(correlation_matrix, mask=mask, annot=correlation_matrix.rank(axis="columns"), cmap="autumn", cbar=ax_heatmap)
    ax_heatmap.set_title("Heatmap of correlation (by rank)")

### TRAIN TEST SPLIT ###
def test_train_splitter(dfs,  test_size=0.2):
    # train data - 60% / 74%
    # test data - 20% / 13%
    # validation data - 20% / 13%
    # train, validate, test = np.split(df.sample(frac=1), [int(tr * len(df)), int((tr+va) * len(df))])
    df = dfs["selected_columns_dataset"]
    X = df.drop(['fraudulent'],axis=1)
    Y = df["fraudulent"]

    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=8)
    x_validation_data = []
    y_validation_data = []

    # Use the same function above for the validation set
    # x_train_data, x_validation_data, y_train_data, y_validation_data = train_test_split(x_train0_data, y_train0_data, test_size=0.25, random_state=8)
    print("-*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-")
    print("Train Data =",len(x_train_data) , "; Test Data =" ,len(x_test_data), "; Validation Data =", len(x_validation_data))

    tvt_data = { "x_train_data": x_train_data, "x_validation_data": x_validation_data, "x_test_data": x_test_data, "y_train_data": y_train_data, "y_validation_data": y_validation_data, "y_test_data": y_test_data }

    total_data = len(x_train_data) + len(x_validation_data) + len(x_test_data)
    print("Split Train - Validation - Test: ", round(len(x_train_data)/total_data, 4), '-', round(len(x_validation_data)/total_data, 4), '-', round(len(x_test_data)/total_data, 4))

    return tvt_data

def knn_classifier(data):
    print("\n-- KNN Classifier ---\n")

    x_train_data = data['x_train_data']
    y_train_data = data['y_train_data']
    x_validation_data = data['x_validation_data']
    y_validation_data = data['y_validation_data']
    x_test_data = data['x_test_data']
    y_test_data = data['y_test_data']

    n = [1, 3, 5, 10]
    all_metri = pd.DataFrame()
    for k in n:
        print('* '+str(k)+"NN *")
        model = KNeighborsClassifier(n_neighbors=k).fit(x_train_data,y_train_data)

        # y_pred = model.predict(x_validation_data)
        # accuracy = metrics.accuracy_score(y_validation_data, y_pred)
        # print("Accuracy V=", accuracy)
        # recall = metrics.recall_score(y_validation_data, y_pred)
        # print("Recall V=", recall)
        # precision = metrics.precision_score(y_validation_data, y_pred)
        # print("Precision V=", precision)

        y_pred = model.predict(x_test_data)
        accuracy = metrics.accuracy_score(y_test_data, y_pred)
        print("Accuracy=", accuracy)
        recall = metrics.recall_score(y_test_data, y_pred)
        print("Recall=", recall)
        precision = metrics.precision_score(y_test_data, y_pred)
        print("Precision=", precision)
        data = {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision}
        metri = pd.DataFrame(data=data, index=[str(k)+"NN"])
        all_metri = pd.concat([all_metri, metri])
        
        # metrics.plot_confusion_matrix(model, x_test_data, y_test_data)
    print(all_metri)

if __name__ == "__main__":
    full_dataset = read_data(filename)
    intial_data_visualization(full_dataset)
    dfs = preprocess_data(full_dataset)
    heatmap_correlation(dfs['selected_columns_dataset'])
    data = test_train_splitter(dfs)
    knn_classifier(data)

    plt.show()