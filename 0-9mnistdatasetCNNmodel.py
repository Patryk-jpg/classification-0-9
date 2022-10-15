
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.func_graph import flatten
import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
from random import randint
(train_data, train_data_labels),(test_data,test_data_labels)= tf.keras.datasets.mnist.load_data(path = 'mnist.npz')
#wczytaj dane ze zbioru danych do 2 tupli, 1 na dane i nazwy klas do treningu a drugi do testowania

n = train_data.shape

train_data = train_data / 255.0
test_data =  test_data / 255.0

n = train_data.shape
print(n) #widzimy ze wysokosc i dlugosc obrazkow to 28x28
#zmienne do modelu
IMG_WIDTH = 28
IMG_HEIGHT = 28
colorchannels = 1 #obrazki ssa czarno-biale wiec jest tylko jeden, w wypadku RGb bylyby 3
class_names = ['0', '1', ' 2', '3', '4', '5', '6', '7', '8', '9']
#model

def Creation():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape = (IMG_WIDTH,IMG_HEIGHT,1) , activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(10,activation= 'softmax')
    ])
    model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics= ['accuracy'])
    return model

model  = Creation()
def trainModel(model):
    filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.summary()

    model.fit(train_data, train_data_labels,validation_split=0.2, batch_size = 64, epochs = 10, callbacks = callbacks_list)
    
# Loads the weights
model.load_weights('weights-improvement-08-0.99.hdf5')

# Re-evaluate the model
loss, acc = model.evaluate(test_data, test_data_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


prediction = model.predict(np.array([test_data[0]]))
print(prediction)
window = tk.Tk()
window.geometry('800x600')
test_data = test_data*255.0
def predict_number(a):
    prediction = model.predict(np.array([test_data[a]]))
    prediction_class = class_names[np.argmax(prediction)]
    return prediction_class
def pokaz():
    a = randint(0,len(test_data))
    photo = np.array(test_data[a]) #konwertuje na numpy array
    photo = Image.fromarray(photo) #konwertuje Numpy array z obrazkiem pojedynczym na PIL obiekt 
    photo = photo.resize((200,200))
    photo = ImageTk.PhotoImage(photo) #konwertuje PIL obiekt na taki ktory da sie dać do tkintera
    obrazek = Label(window, image = photo)
    obrazek.image = photo
    obrazek.grid(column= 1,row= 0)
    p = predict_number(a)
    correctp = test_data_labels[a]
    Textlabel['text'] = 'Ta liczba to', correctp, 'Według programu jest to: ', p
Przycisk = Button(window, text = 'generate' , command = pokaz)
Przycisk.grid(column=0,row=0)
Textlabel = Label(window)
Textlabel.grid(column=1,row=1)
window.mainloop()