# Imports
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models


# Turn off GPU
tf.config.set_visible_devices([], "GPU")


# Read in the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--c_input", type=int, default=2)
parser.add_argument("--c_hidden", type=int, default=2)
parser.add_argument("--c_output", type=int, default=2)

parser.add_argument("--k_input", type=float, default=0.1)
parser.add_argument("--k_hidden", type=float, default=0.1)
parser.add_argument("--k_output", type=float, default=0.1)

args = parser.parse_args()



# Create the dataset
n = 5000
X = np.linspace(-5, 5, n)
Y = np.cosh(X)


# Set the default parameters
val_split = 0.2
n_epochs = 200


# Create the model
inputs = layers.Input(shape=(1,))

layer_1 = layers.Dense(args.c_input, activation="relu")(inputs)
layer_2 = layers.Dropout(args.k_input)(layer_1)

layer_3 = layers.Dense(args.c_hidden, activation="relu")(layer_2)
layer_4 = layers.Dropout(args.k_hidden)(layer_3)

layer_5 = layers.Dense(args.c_output, activation="relu")(layer_4)
layer_6 = layers.Dropout(args.k_output)(layer_5)

outputs = layers.Dense(1, activation="linear")(layer_6)


model = models.Model(inputs=inputs, outputs=outputs, name="NN_cosh")


# Compile and train the model
model.compile(loss="mse", metrics=["mae"],
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01))
model.fit(x=X, y=Y, validation_split=0.2, epochs=200, shuffle=True, verbose=0)


# Evaluate the model
res = model.evaluate(X, Y, verbose=0)

print(res[1])