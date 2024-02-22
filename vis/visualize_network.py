import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Input,Flatten, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
#from keras_visualizer import visualizer
import visualkeras

# Assuming self.in_size, self.n_out_class, self.n_feat are defined
# Define the input shape (adjust as needed based on your input size)

# Define the Keras model
model = Sequential()
model.add(Input(shape=(5,1,288))) # Input tensor
# Block 1
model.add(Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=4, strides=4))

# Block 2
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=4, strides=4))

# Block 3
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=4, strides=4))

# Block 4
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=4, strides=4))

# Block 5
model.add(Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'))

model.add(GlobalMaxPooling1D())

# Block 6
model.add(Dense(512, activation='relu'))

# l7
model.add(Dense(3))


visualkeras.layered_view(model)