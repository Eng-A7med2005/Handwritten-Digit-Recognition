import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib import style
style.use("dark_background")

df = tf.keras.datasets.mnist
(train_images, train_labels) , (test_images , test_labels) = df.load_data()

class_names = ['0','1','2','3','4','5','6','7','8','9']

train_images = tf.keras.utils.normalize(train_images, axis= 1)          
test_images = tf.keras.utils.normalize(test_images, axis= 1)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)


history = model.fit(train_images,train_labels, epochs=10)
history.history.keys()

test_loss , test_acc = model.evaluate(test_images , test_labels , verbose=1)
print('Test accuracy:', test_acc)


pred = int(input("Choose a number honey : "))
label = str(class_names[test_labels[pred]])
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[pred])])
guess = class_names[np.argmax(predictions[pred])]


plt.figure(1)
plt.imshow(test_images[pred] , cmap='gray')
plt.title("Expected: " + label)
plt.xlabel("Guess: " + guess)
plt.grid(False)

plt.figure(2)
plt.plot(history.history['accuracy'] , color='red')
plt.title("Model Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train'], loc='upper left')

plt.figure(3)
plt.plot(history.history['loss'],  color='red')
plt.title("Model LOSS")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['train'], loc='upper left')
plt.show()


model.save('epic_num_reader.h')
