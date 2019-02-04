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
from sklearn.model_selection import StratifiedKFold, cross_val_score

# loading data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshaping, normalizing and one hot encode outputs
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
y_train_KFold = y_train
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#StratifiedKFold
skf = StratifiedKFold(n_splits=5)

k_history = []
scores = []

for train_index, test_index in skf.split(x_train, y_train_KFold):
    model = Sequential()
    model.add(Conv2D(4, (5,5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    #compilation
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
    #train on the k-partition
    temp_train = x_train[train_index]
    temp_target = y_train[train_index]
    temp_test = x_train[test_index]
    temp_test_target = y_train[test_index]
    k_history.append(model.fit(temp_train,temp_target, batch_size = 128, epochs = 10))
    scores.append(model.evaluate(temp_test, temp_test_target, batch_size = 128))
    

for i in range(0, 5):
    plt.plot(k_history[i].history['acc'])

plt.title('Accuracy of K test')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['1st', '2nd','3rd','4th','5th'], loc='upper left')
plt.savefig('k_fold_validation.png')
plt.show()
plt.clf()


print("Scores of tests:")
print(scores)
