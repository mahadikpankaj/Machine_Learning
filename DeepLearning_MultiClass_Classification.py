#!/usr/bin/env python
# coding: utf-8

# In[1]:


def plot_decision_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 500)
    vticks = np.linspace(bmin, bmax, 500)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    # make prediction with the model and reshape the output so contourf can plot it
    y_pred_enc = model.predict(ab)

    y_pred_enc = np.round(y_pred_enc)
    y_pred=np.array([np.argmax(y, axis=None, out=None) for y in y_pred_enc])
    Z = y_pred.reshape(aa.shape)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # plot the contour
    cntr1 = ax1.contourf(aa, bb, Z, cmap='Pastel1', alpha=0.8)
#    ax1.clabel(cntr1, inline=True, fontsize=10, use_clabeltext=True, colors='b')
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2', ec='darkgrey', marker='o', s=50)
    return plt


# In[2]:


#!python -m pip install --upgrade tensorflow keras 

from sklearn.datasets import make_blobs, make_circles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

target_labels_count = 5

X, y = make_blobs(n_samples=target_labels_count*500, centers=target_labels_count, n_features=2, random_state=42)
#X, y = make_circles(n_samples=target_labels_count*500, factor=.6, noise=0.1, random_state=42)

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if target_labels_count > 1:
    y_train_enc = to_categorical(y_train)
    y_test_enc = to_categorical(y_test)
else:
    y_train_enc = y_train
    y_test_enc = y_test

activation_f = 'softmax'
#activation_f = 'sigmoid'
#activation_f = 'tanh'

loss_f = 'categorical_crossentropy'
#loss_f = 'binary_crossentropy'

model = Sequential()
model.add(Dense(9, activation='relu', input_dim=2))
model.add(Dense(7, activation='tanh'))
model.add(Dense(target_labels_count, activation=activation_f))

# Compile the model
#model.compile(optimizer='adam', loss=loss_f, metrics=['accuracy'])

from keras.optimizers import Adam
model.compile(Adam(learning_rate=0.05), 'binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=500, mode='max')]
model.fit(X_train, y_train_enc, epochs=200, verbose=0, callbacks=my_callbacks, validation_data=(X_test, y_test_enc))
eval_result = model.evaluate(X_test, y_test_enc)
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])

y_pred = model.predict(X_test)
y_pred_enc = np.round(y_pred)
y_pred=[np.argmax(y, axis=None, out=None) for y in y_pred_enc]

plot_decision_boundary(model, X, y).show()


# In[3]:


#!python -m pip install --upgrade tensorflow keras 

from sklearn.datasets import make_blobs, make_circles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

target_labels_count = 1

#X, y = make_blobs(n_samples=target_labels_count*500, centers=target_labels_count, n_features=2, random_state=42)
X, y = make_circles(n_samples=target_labels_count*500, factor=.6, noise=0.1, random_state=42)

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if target_labels_count > 1:
    y_train_enc = to_categorical(y_train)
    y_test_enc = to_categorical(y_test)
else:
    y_train_enc = y_train
    y_test_enc = y_test

#activation_f = 'softmax'
activation_f = 'sigmoid'
#activation_f = 'tanh'

#loss_f = 'categorical_crossentropy'
loss_f = 'binary_crossentropy'

model = Sequential()
model.add(Dense(9, activation='relu', input_dim=2))
model.add(Dense(7, activation='tanh'))
model.add(Dense(target_labels_count, activation=activation_f))

# Compile the model
#model.compile(optimizer='adam', loss=loss_f, metrics=['accuracy'])

from keras.optimizers import Adam
model.compile(Adam(learning_rate=0.05), 'binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=500, mode='max')]
model.fit(X_train, y_train_enc, epochs=200, verbose=0, callbacks=my_callbacks, validation_data=(X_test, y_test_enc))
eval_result = model.evaluate(X_test, y_test_enc)
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])

y_pred = model.predict(X_test)
y_pred_enc = np.round(y_pred)
y_pred=[np.argmax(y, axis=None, out=None) for y in y_pred_enc]

plot_decision_boundary(model, X, y).show()
