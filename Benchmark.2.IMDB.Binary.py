from tensorflow.keras.datasets import imdb
import numpy as np
import time

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float16')
y_test = np.asarray(test_labels).astype('float16')

from tensorflow.keras import models, layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# from keras import optimizers, losses, metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy(),
# metrics=[metrics.binary_accuracy()])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

start_fit = time.time()

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
end_fit = time.time()

Results = model.evaluate(x_test,y_test)
print("Test Accuracy: ", Results[1]*100, "%")

print("--- %s seconds ---" % (end_fit - start_fit), " fit time.")
