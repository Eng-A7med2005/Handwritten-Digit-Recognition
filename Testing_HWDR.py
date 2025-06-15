import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib import style
from tensorflow.keras.models import load_model
style.use("dark_background")

df = tf.keras.datasets.mnist
(train_images, train_labels) , (test_images , test_labels) = df.load_data()

class_names = ['0','1','2','3','4','5','6','7','8','9']

train_images = tf.keras.utils.normalize(train_images, axis= 1)
test_images = tf.keras.utils.normalize(test_images, axis= 1)

model = load_model(r"E:\Projects\Uneeq\epic_num_reader.h5")

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
plt.show()

def on_key(event):
    if event.key == 'q':
        plt.close()
