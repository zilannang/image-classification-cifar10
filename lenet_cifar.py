import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.models import load_model
from numpy.testing import assert_allclose
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from  numpy  import  loadtxt 
from scipy import interp
from itertools import cycle
from keras.layers import AveragePooling2D

batch_size = 8 
num_classes = 10
epochs = 10
input_shape =(32, 32, 3)
n_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#model.load("model_weights.h5")

#Matrisleri normalize etmek
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Verileri kategorik yapıya çevirme
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

#Ağ oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu' ,input_shape=input_shape))
model.add(AveragePooling2D((2, 2), padding='same'))
#model.add(Dropout(0.1))
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(AveragePooling2D((2, 2), padding='same'))
#model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

#Derleme
model.compile(optimizer='adam', #lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
#Gereksiz epoch yapmama
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='model_weights.h5', monitor='val_loss', save_best_only=True)]
'''
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen_generator = datagen.flow(x_train[:40000], y_train[:40000], batch_size=batch_size)

#Modeli eğit
history = model.fit_generator(datagen.flow(x_train, y_train, 
              batch_size=batch_size),
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
              
'''
#modeli değerlendir
scores=model.evaluate(x_test, y_test,verbose=0)
print('Baseline error:%.2f'%(1-scores[1]))

print("Accuracy =%.2f"%scores[1])
'''
Y_pred = model.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

#confusion matrix
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

model_json = model.to_json()
open('cifar10_model.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)
 
print(history.history.keys())
 

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()