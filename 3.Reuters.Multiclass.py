from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import time

# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Temp fix allow_pickle = False
import numpy as np

"""# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# restore np.load for future normal usage
np.load = np_load_old"""

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# def to_one_hot(labels,dimension=46):
#    results = np.zeros((len(labels), dimension))
#    for i, label in enumerate(labels):
#        results[i,label] = 1.
#     return results
#
# y_train_labels = to_one_hot(train_labels)
# y_test_labels = to_one_hot(test_labels)

y_train_labels = to_categorical(train_labels)
y_test_labels = to_categorical(test_labels)

model = models.Sequential()
# Using 64 dimensional data instead of 16 for greater granularity
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# Use 46 because there are 46 types of Newspaper
# Thus the output is a 46 vector of probabilities for each category
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train_labels[:1000]
partial_y_train = y_train_labels[1000:]

start_fit = time.time()
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
end_fit = time.time()

import PlotLossAcc
PlotLossAcc.myplot(history)

Results = model.evaluate(x_test, y_test_labels)

model = models.Sequential()
# Using 64 dimensional data instead of 16 for greater granularity
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# Use 46 because there are 46 types of Newspaper
# Thus the output is a 46 vector of probabilities for each category
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(partial_x_train,partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

ImprovedResults = model.evaluate(x_test, y_test_labels)

print("Test Accuracy (NEW): ", ImprovedResults[1]*100, "%")

print("Test Accuracy (OLD): ", Results[1]*100, "%")

print("--- %s seconds ---" % (end_fit - start_fit), " fit time.")
