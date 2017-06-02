from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

# Dropout keep probability.
drop_prob = 0.1
in_shape = (1,1,1)

# Model definition.
model = Sequential()
model.add(Conv2D(32, 5, input_shape=(in_shape)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 5))
model.add(MaxPooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(drop_prob))
model.add(Dense(2, activation='softmax'))

# Model compilation.
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training.
## Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

## Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)