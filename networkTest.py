import keras
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTest = tf.keras.utils.normalize(xTest, axis = 1)
for img in range(len(xTest)):
    for row in range(28):
        for col in range(28):
            if xTest[img][row][col] != 0:
                xTest[img][row][col] = 1

model = tf.keras.models.load_model('num-predicter.model')
print(len(xTest))
predictions = model.predict(xTest)

count = 0

for i in range(len(predictions)):
    guess = np.argmax(predictions[i])
    actual = yTest[i]
    # print("The prediction value by the network is ", guess)
    # print("The actual label for the image is ", actual)
    if guess == actual:
        count += 1
    # plt.imshow(xTest[i], cmap=plt.cm.binary)
    # plt.show()

print("The program got ", count, " correct out of ", len(xTest))
print("Accuracy is", count/len(xTest))