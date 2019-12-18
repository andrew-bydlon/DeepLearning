import numpy as np

# Hold-Out

num_validation_samples = 10000

np.random.shuffle(data)

validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

training_data = data[:]
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# K-fold CrossVal

k = 4
num_validation_samples = len(data) //4

np.random.shuffle(data)

validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples*fold:num_validation_samples*(fold+1)]
    training_data = data[:num_validation_samples*fold]+data[num_validation_samples*(fold+1):]

    model.get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)
