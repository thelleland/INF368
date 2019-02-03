import numpy
import json
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.optimizers import SGD
from matplotlib import pyplot as plt

# loading data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshaping, normalizing and one hot encode outputs
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

model = Sequential()
model.add(Conv2D(4, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
#compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

model.summary()

#Loading mlp model
with open("json_string.json", "r") as f:
    json_string = f.read()
    
model_mlp = model_from_json(json_string)

#Had to compile loaded model in order to evaluate
model_mlp.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
model_mlp.load_weights("my_model_weights.h5")

#Loading history of mlp model
with open("history_dict.json", "r") as f:
    history_dict = f.read()

history_mlp = json.loads(history_dict)

# Training cnn model
history = model.fit(x_train,y_train, validation_split = 0.0833, batch_size = 128, epochs = 10)
history_cnn = history.history


# Plotting training accuracy curves
plt.plot(history_cnn['acc'])
plt.plot(history_mlp['acc'])
plt.title('Comparison of training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train cnn', 'Train mlp'], loc='upper left')
plt.savefig('comp_train_acc.png')
plt.show()
plt.clf()

#Plotting validation accuracy curves 
plt.plot(history_cnn['val_acc'])
plt.plot(history_mlp['val_acc'])
plt.title('Comparison of validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Val cnn', 'Val mlp'], loc='upper left')
plt.savefig('comp_val_acc.png')
plt.show()
plt.clf()
         
#Plotting of training loss
plt.plot(history_cnn['loss'])
plt.plot(history_mlp['loss'])
plt.title('Comparison of model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['cnn loss','mlp loss'], loc='upper left')
plt.savefig('comp_train_loss.png')
plt.show()
plt.clf()

# Plotting of validation loss
plt.plot(history_cnn['val_loss'])
plt.plot(history_mlp['val_loss'])
plt.title('Comparison of validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['cnn loss', 'mlp loss'], loc='upper left')
plt.savefig('comp_val_loss.png')
plt.show()
plt.clf()

#I reshaped the first model into a 2D vector, must reshape here as well
score_mlp = model_mlp.evaluate(x_test.reshape(10000,784), y_test, batch_size = 128)
score_cnn = model.evaluate(x_test, y_test, batch_size = 128)

print("Score of first model:")
print(score_mlp)
print("Score of cnn: ")
print(score_cnn)

temporal_mean_acc_mlp = sum(history_mlp['val_acc']) / len(history['val_acc'])
temporal_mean_acc_cnn = sum(history_cnn['val_acc']) / len(history['val_acc'])
print("Temporal accuracy of first model validation accuracy:")
print(temporal_mean_acc_mlp)
print("Temporal accuracy of cnn model validation accuracy:")
print(temporal_mean_acc_cnn)

