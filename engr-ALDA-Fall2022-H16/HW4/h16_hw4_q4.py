# Importing All Required Libraries
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")

# Setting random seed = 2022
os.environ['PYTHONHASHSEED'] = '2022'
random.seed(2022)
np.random.seed(2022)
tf.random.set_seed(2022)

# Loading train, validate and test datasets
Ann_train = pd.read_csv("ann_2022/train_data_2022.csv")
Ann_validate = pd.read_csv("ann_2022/val_data_2022.csv")
Ann_test = pd.read_csv("ann_2022/test_data_2022.csv")

# Creating train, validate, test dataframes for attributes and labels
attr_train = Ann_train.shape[1] - 1
attr_validate = Ann_validate.shape[1] - 1
attr_test = Ann_test.shape[1] - 1

# training dataset
x_train = Ann_train.iloc[:,0:attr_train]
y_train = Ann_train.iloc[:,attr_train]
# validation dataset
x_validate = Ann_validate.iloc[:,0:attr_validate]
y_validate = Ann_validate.iloc[:,attr_validate]
# testing dataset
x_test = Ann_test.iloc[:,0:attr_test]
y_test = Ann_test.iloc[:,attr_test]

# Training models on Training dataset and Evaluating models on Validation dataset
neurons = [4,16,32,64]
train_results = []
val_results = []
model_list = []

for n in neurons:
    # Training model
    model = Sequential()
    print(f"***********Training model with {n} Neurons in hidden layer**********")
    model.add(Dense(n, input_shape=(attr_train,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # compile the keras model
    train_value = model.fit(x_train, y_train, epochs=5, batch_size=10)
    train_results.append(train_value)
    print("\n")
    model_list.append(model)

    # Validating
    print (f"***********Evaluating model with {n} Neurons in hidden layer**********")
    accuracy = model.evaluate(x_validate, y_validate, batch_size=10)
    val_results.append(accuracy[1])

# Ploting results against training and validating models with respect to their hidden neurons
fig1, ax1 = plt.subplots(1,1, figsize=(18,15))
tr_line, = ax1.plot(neurons, [train_results[i].history['accuracy'][-1] for i in range(len(neurons))], color="orange", linewidth=2, label="Training Accuracy")
ax1.plot(neurons, [train_results[i].history['accuracy'][-1] for i in range(len(neurons))], "ro")
vl_line, = ax1.plot(neurons, [val_results[i] for i in range(len(neurons))], 'b-', linewidth=2, label="Validation Accuracy")
ax1.plot(neurons, [val_results[i] for i in range(len(neurons))], 'co')
ax1.set_title('Hidden Number of Neurons vs Accuracy')
ax1.set_xlabel('Hidden Number of Neurons')
ax1.set_ylabel('Accuracy')
for i in range(len(neurons)):
    ax1.annotate('{:.4f}'.format(train_results[i].history['accuracy'][-1]), (neurons[i], train_results[i].history['accuracy'][-1]))
    ax1.annotate('{:.4f}'.format(val_results[i]), (neurons[i], val_results[i]))
ax1.legend(handles=[tr_line, vl_line])
plt.show()

# Testing model on testing dataset
optimal_n_idx = 2
print (f"***********testing model with 32 Neurons in hidden layer**********")
test_accuracy = model_list[optimal_n_idx].evaluate(x_test, y_test, batch_size=10)
print(f"loss: {test_accuracy[0]} - accuracy: {test_accuracy[1]}")

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5))
# ax[0].set_title('Hidden Number of Neurons vs Training Accuracy')
# ax[0].set_ylabel('Training Accuracy')
# ax[0].set_xlabel('Hidden Number of Neurons')
# ax[0].bar(neurons, [train_results[i].history['accuracy'][-1] for i in range(len(neurons))], width=4, color='orange')
# for i in range(len(neurons)):
#     ax[0].annotate('{:.4f}'.format(train_results[i].history['accuracy'][-1]), (neurons[i], train_results[i].history['accuracy'][-1]))

# ax[1].set_title('Hidden Number of Neurons vs Validation Accuracy')
# ax[1].set_ylabel('Validation Accuracy')
# ax[1].set_xlabel('Hidden Number of Neurons')
# ax[1].bar(neurons, [val_results[i] for i in range(len(neurons))], width=5, color='green')
# for i in range(len(neurons)):
#     ax[1].annotate('{:.4f}'.format(val_results[i]), (neurons[i]-0.4, val_results[i]-0.05))


##### OUTPUT #####
"""
***********Training model with 4 Neurons in hidden layer**********
Epoch 1/5
100/100 [==============================] - 0s 1ms/step - loss: 0.6694 - accuracy: 0.5890
Epoch 2/5
100/100 [==============================] - 0s 1ms/step - loss: 0.6312 - accuracy: 0.6880
Epoch 3/5
100/100 [==============================] - 0s 1ms/step - loss: 0.5839 - accuracy: 0.7700
Epoch 4/5
100/100 [==============================] - 0s 1ms/step - loss: 0.5270 - accuracy: 0.8070
Epoch 5/5
100/100 [==============================] - 0s 1ms/step - loss: 0.4712 - accuracy: 0.8310


***********Evaluating model with 4 Neurons in hidden layer**********
25/25 [==============================] - 0s 1ms/step - loss: 0.4471 - accuracy: 0.8640
***********Training model with 16 Neurons in hidden layer**********
Epoch 1/5
100/100 [==============================] - 0s 1ms/step - loss: 0.6750 - accuracy: 0.5870
Epoch 2/5
100/100 [==============================] - 0s 2ms/step - loss: 0.5863 - accuracy: 0.7480
Epoch 3/5
100/100 [==============================] - 0s 1ms/step - loss: 0.4897 - accuracy: 0.8360
Epoch 4/5
100/100 [==============================] - 0s 1ms/step - loss: 0.4047 - accuracy: 0.8660
Epoch 5/5
100/100 [==============================] - 0s 1ms/step - loss: 0.3479 - accuracy: 0.8770


***********Evaluating model with 16 Neurons in hidden layer**********
25/25 [==============================] - 0s 2ms/step - loss: 0.3702 - accuracy: 0.8360
***********Training model with 32 Neurons in hidden layer**********
Epoch 1/5
100/100 [==============================] - 1s 1ms/step - loss: 0.6525 - accuracy: 0.6260
Epoch 2/5
100/100 [==============================] - 0s 1ms/step - loss: 0.5223 - accuracy: 0.8200
Epoch 3/5
100/100 [==============================] - 0s 1ms/step - loss: 0.4112 - accuracy: 0.8730
Epoch 4/5
100/100 [==============================] - 0s 1ms/step - loss: 0.3419 - accuracy: 0.8900
Epoch 5/5
100/100 [==============================] - 0s 1ms/step - loss: 0.3046 - accuracy: 0.8890


***********Evaluating model with 32 Neurons in hidden layer**********
25/25 [==============================] - 0s 1ms/step - loss: 0.3519 - accuracy: 0.8600
***********Training model with 64 Neurons in hidden layer**********
Epoch 1/5
100/100 [==============================] - 1s 1ms/step - loss: 0.6295 - accuracy: 0.6820
Epoch 2/5
100/100 [==============================] - 0s 2ms/step - loss: 0.4642 - accuracy: 0.8520
Epoch 3/5
100/100 [==============================] - 0s 1ms/step - loss: 0.3484 - accuracy: 0.8830
Epoch 4/5
100/100 [==============================] - 0s 1ms/step - loss: 0.2976 - accuracy: 0.8880
Epoch 5/5
100/100 [==============================] - 0s 1ms/step - loss: 0.2742 - accuracy: 0.8910


***********Evaluating model with 64 Neurons in hidden layer**********
25/25 [==============================] - 0s 1ms/step - loss: 0.3594 - accuracy: 0.8360
"""
