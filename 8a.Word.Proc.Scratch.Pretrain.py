import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

imdb_dir = '/home/andrew/PycharmProjects/DeepLearning/IMDB'
train_dir = os.path.join(imdb_dir, 'train')
labels, texts = [], []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

xtrain = data[:training_samples]
ytrain = labels[:training_samples]
xval = data[training_samples: training_samples + validation_samples]
yval = labels[training_samples: training_samples + validation_samples]

glove = '/home/andrew/PycharmProjects/DeepLearning/Glove'
embeddingsindex = {}
f = open(os.path.join(glove, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddingsindex[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddingsindex))

embeddingDim = 100

embdeddingMatrix = np.zeros((max_words, embeddingDim))
for word, i in word_index.items():
    if i < max_words:
        embeddingVector = embeddingsindex.get(word)
        if embeddingVector is not None:
            embdeddingMatrix[i] = embeddingVector

model = Sequential()
model.add(Embedding(max_words, embeddingDim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.layers[0].set_weights([embdeddingMatrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xval, yval))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()