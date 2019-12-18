import tensorflow as tf
import time

# tf.device('/gpu:0')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() # Assign variable tuples to the data

network = tf.keras.models.Sequential()  # Create sequential network

network.add(tf.keras.layers.Dense(512, activation='relu',input_shape=(28*28,)))  # layer of image data
network.add(tf.keras.layers.Dense(10, activation='softmax'))  # layer of 10 probability scores
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float16')/255
test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float16')/255
# transforms the data from int8 values between 0 and 255 to float32 values between 0 and 1

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
# categorically encodes the labels that exist in each data set

start_time = time.time()

# network.fit(train_images, train_labels, epochs=10, batch_size=128) # fits the network to the data
network.fit(train_images, train_labels, epochs=10, batch_size=512) # fits the network to the data
test_loss, test_acc = network.evaluate(test_images, test_labels) # evaluates our network on test_images

print("My Test Accuracy is: ", test_acc*100)
print("--- %s seconds ---" % (time.time() - start_time))