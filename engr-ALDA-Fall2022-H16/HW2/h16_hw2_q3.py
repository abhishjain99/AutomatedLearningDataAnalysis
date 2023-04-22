# pandas Documentation - https://pandas.pydata.org/docs/
# Confusion Matrix - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
# Evaluation Metrics - https://en.wikipedia.org/wiki/Confusion_matrix | https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

print("\n(a) ----- Optimistic Errors -----\n")
# iii. performance on the following five evaluation metrics: Accuracy, Recall (Sensitivity), Precision, Specificity, and F1 Measure. Treat “On Time” as a positive outcome and “Delay” as a negative outcome.
bus_dataset = pd.read_csv("BusDelay.csv")
y_true_o = bus_dataset['Outcome'].to_list()
y_pred_o = bus_dataset['DT Outcome'].to_list()
tp_o, fn_o, fp_o, tn_o = metrics.confusion_matrix(y_true_o, y_pred_o, labels=['On Time', 'Delay']).ravel()
# print(tp_o, fn_o, fp_o, tn_o)
# [[5 3]
#  [2 6]]
# prediction - col, actual - row, confusion matrix:
# [[tp fn]
#  [fp tn]]

accuracy = (tp_o+tn_o)/(tp_o+fn_o+fp_o+tn_o)
# accuracy = metrics.accuracy_score(y_true_o, y_pred_o)
recall = tp_o/(tp_o+fn_o)
# recall = metrics.recall_score(y_true_o, y_pred_o, pos_label='On Time')
precision = tp_o/(tp_o+fp_o)
# precision = metrics.precision_score(y_true_o, y_pred_o, pos_label='On Time')
specificity = tn_o/(tn_o+fp_o)
f1_measure = 2*tp_o/(2*tp_o+fn_o+fp_o)
# f1_measure = metrics.f1_score(y_true_o, y_pred_o, pos_label='On Time')
print("a. iii. Performance--\nAccuracy:", accuracy, "\nRecall:", recall, "\nPrecision:", precision, "\nSpecificity:", specificity, "\nF1 Measure", f1_measure)

print("\n(b) ----- Pessimistic Errors -----\n")
# Classified Dataset is used to evaluate the metrics
bus_classified_dataset = pd.read_csv("ClassifiedDataset.csv")
y_true_p = bus_classified_dataset['Outcome'].to_list()
y_pred_p = bus_classified_dataset['DT Outcome'].to_list()
tp_p, fn_p, fp_p, tn_p = metrics.confusion_matrix(y_true_p, y_pred_p, labels=['On Time', 'Delay']).ravel()
# print(tp_p, fn_p, fp_p, tn_p)

classified_dc_accuracy = (tp_p+tn_p)/(tp_p+fn_p+fp_p+tn_p)
# classified_dc_accuracy = metrics.accuracy_score(y_true_p, y_pred_p)
classified_dc_recall = tp_p/(tp_p+fn_p)
# classified_dc_recall = metrics.recall_score(y_true_p, y_pred_p, pos_label='On Time')
classified_dc_precision = tp_p/(tp_p+fp_p)
# classified_dc_precision = metrics.precision_score(y_true_p, y_pred_p, pos_label='On Time')
classified_dc_specificity = tn_p/(tn_p+fp_p)
classified_dc_specificity = tn_p/(tn_p+fp_p)
classified_dc_f1_measure = 2*tp_p/(2*tp_p+fn_p+fp_p)
# classified_dc_f1_measure = metrics.f1_score(y_true_p, y_pred_p, pos_label='On Time')
print("b. iii. Performance--\nAccuracy:", classified_dc_accuracy, "\nRecall:", classified_dc_recall, "\nPrecision:", classified_dc_precision, "\nSpecificity:", classified_dc_specificity, "\nF1 Measure", classified_dc_f1_measure)

##### Output #####
"""
(a) ----- Optimistic Errors -----

a. iii. Performance--
Accuracy: 0.6875 
Recall: 0.625 
Precision: 0.7142857142857143 
Specificity: 0.75 
F1 Measure 0.6666666666666666

(b) ----- Pessimistic Errors -----

b. iii. Performance--
Accuracy: 0.8125 
Recall: 1.0 
Precision: 0.7272727272727273 
Specificity: 0.625 
F1 Measure 0.8421052631578948
"""
