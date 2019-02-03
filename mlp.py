import keras
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
import json

num_classes = 10

#Splitting the data in training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

shape_train = x_train.shape
print("Shape of training set")
print(shape_train)
print("Shape of test set")
shape_test = x_test.shape
print(shape_test)


plt.imshow(x_train[0,:])
plt.show()
plt.imshow(x_train[10000,:])
plt.show()

#Reshaping into proper dimension, changing type and normalizing datasets
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')/256
x_test = x_test.astype('float32')/256


#Transforming labels into class vectors
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Comment on the softmax here!!
model = Sequential()
model.add(Dense(128, input_shape= (784,)))
#model.add(Flatten())
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))


print("\nArchitecture of model: \nInput layer, 784 input nodes, sigmoid activation\nHidden layer, 128 nodes, softmax activation\nOutput layer, 10 output nodes")

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])


# Training model, batch_size = 64
history = model.fit(x_train,y_train, validation_split = 0.0833, batch_size = 64, epochs = 10)
#Plotting accuracy, batch_size = 64
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy, batch size = 64')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('mlp_acc_64.png')
plt.show()


# Training model, batch_size = 128
history = model.fit(x_train,y_train, validation_split = 0.0833, batch_size = 128, epochs = 10)

#Plotting accuracy, batch_size = 128
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy, batch size = 128')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('mlp_acc_128.png')
plt.show()


#Plotting of loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss, batch size = 128')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('mlp_loss_128.png')
plt.show()

#Saving model, weights and history
json_string = model.to_json()
with open("json_string.json", "w") as json_file:
    json_file.write(json_string)
model.save_weights('my_model_weights.h5')
with open("hist_file.json", "w") as f:
    json.dump(history.history, f)

history_dict = history.history
json.dump(history_dict, open("history_dict.json", "w"))
