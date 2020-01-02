import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain = tf.keras.utils.normalize(xTrain, axis = 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)

for img in range(len(xTrain)):
    for row in range(28):
        for col in range(28):
            if xTrain[img][row][col] != 0:
                xTrain[img][row][col] = 1

# The images are of nos in a grayscale channel. So to normalize the image we convert the pixel values to either black or not black.
# ie to 0's or 1's other values are not allowed
# [[0, 0.23, 0.56, 0],
# [0, 0.23, 0.56, 0],
# [0, 0.23, 0.56, 0],
# [0, 0.23, 0.56, 0]]
# The image is a matrix like this with 28 rows and 28 cols.
# We convert it to
# [[0, 1, 1, 0],
# [0, 1, 1, 0],
# [0, 1, 1, 0],
# [0, 1, 1, 0]]

# Displaying one of the images.
# plt.figure(figsize=(15, 15))
# plt.imshow(xTrain[1])
# plt.xlabel("An Example image of a digit")
# plt.colorbar()
# plt.grid(False)
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=10)
model.save('num-predicter.model')
print("Model Saved as num-predicter.model")

