from keras import layers, models, optimizers
import NvidiaMixedPrec
from keras.preprocessing.image import ImageDataGenerator
import os

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=.2, horizontal_flip=True, fill_mode='nearest')

working_dir = '/home/andrew/PycharmProjects/DeepLearning/CatDog/CatDogWorking'

train_dir = os.path.join(working_dir, 'train')
val_dir = os.path.join(working_dir, 'val')
test_dir = os.path.join(working_dir, 'test')

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=val_generator, validation_steps=50)

model.save('/home/andrew/PycharmProjects/DeepLearning/cats.vs.dogs.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label="Training Acc")
plt.plot(epochs, val_acc, 'b', label="Validation Acc")
plt.title('Training/Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title('Training/Validation Loss')
plt.legend()

plt.show()

Results = model.evaluate(test_generator)

print("Accuracy on Test data:", Results[1]*100, "%")
