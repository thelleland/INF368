import keras
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


num_classes = 10

#Splitting the data in training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

shape_train = x_train.shape
print(shape_train)
shape_test = x_test.shape
print(shape_test)
model = Sequential

plt.imshow(x_train[0,:])
plt.show()
plt.imshow(x_train[10000,:])
plt.show()

#Reshaping into proper dimension, changing type and normalizing datasets
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')/256
x_test = x_test.astype('float32')/256


#Transforming labels into class vectors
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Creating model
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))




print("\nArchitecture of model: \nInput layer, 784 input nodes, sigmoid activation\nHidden layer, 32 nodes, softmax activation\nOutput layer, 10 output nodes")

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])



history = model.fit(x_train,y_train, validation_split = 0.0833, batch_size = 64, epochs = 10)

#Visualization of error with batch size 32
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy, batch size = 32')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualization of error with batch size 64
history = model.fit(x_train,y_train, validation_split = 0.0833, batch_size = 128, epochs = 10)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy, batch size = 64')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualization of loss with batch size 62
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss, batch size = 64')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Savign model and weights
json_string = model.to_json()
model.save_weights('my_model_weights.h5')
