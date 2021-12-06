import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2 as cv


def define_neural_network(nb_filters, input_shape, kernel_size, pool_size, dense_layer_size):
    my_model = Sequential()
    my_model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, activation=relu))
    my_model.add(Conv2D(2 * nb_filters, kernel_size, activation=relu))
    my_model.add(MaxPool2D(pool_size=pool_size))
    my_model.add(Conv2D(4 * nb_filters, kernel_size, activation=relu))
    my_model.add(MaxPool2D(pool_size=pool_size))
    # my_model.add(Conv2D(8 * nb_filters, kernel_size, activation=relu))
    # my_model.add(MaxPool2D(pool_size=pool_size))
    my_model.add(Flatten())
    my_model.add(Dense(dense_layer_size, activation=relu))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(10, activation=softmax))
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(my_model.summary())
    """
    # lenet
    my_model = Sequential()
    my_model.add(Conv2D(6, 5, padding='valid', input_shape=input_shape, activation=relu))
    my_model.add(MaxPool2D(pool_size=2))
    my_model.add(Conv2D(16, 5, activation=relu))
    my_model.add(MaxPool2D(pool_size=2))
    my_model.add(Flatten())
    my_model.add(Dense(120, activation=relu))
    my_model.add(Dense(84, activation=relu))
    my_model.add(Dense(10, activation=softmax))
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(my_model.summary())
    """
    return my_model


def get_number(path):
    start = path.rfind('/')
    number = int(path[start+1:])
    return number


def load_european_digits_data(path):
    x = []
    y = []
    for root, dirs, files in os.walk(path):
        if not dirs:
            nb = get_number(root)
            for file in files:
                path_img = root + '/' + file
                img = 255 - cv.imread(path_img, cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (28, 28))
                x.append(img)
                y.append(nb)
    return np.array(x), np.array(y)


def load_train_test_data(path_to_european_data):
    (x_mnist, y_mnist), (_, _) = tf.keras.datasets.mnist.load_data()
    x_mnist = x_mnist[0:13000]
    y_mnist = y_mnist[0:13000]
    x_european, y_european = load_european_digits_data(path_to_european_data)
    x = np.append(x_mnist, x_european, axis=0)
    y = np.append(y_mnist, y_european, axis=0)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test


# ----------------- data ---------------------
x_train, y_train, x_test, y_test = load_train_test_data('EuropeanDigits')


# ----------------- model ---------------------
train = True
modelname = 'NumberClassifierLarge'
if train:
    nb_filters = 16
    kernel_size = 5
    pool_size = 2
    input_shape = (28, 28, 1)
    dense_layer_size = 128
    my_model = define_neural_network(nb_filters=nb_filters,
                                     input_shape=input_shape,
                                     kernel_size=kernel_size,
                                     pool_size=pool_size,
                                     dense_layer_size=dense_layer_size)
    batch_size = 1000
    nb_epochs = 25
    my_model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs)
    my_model.save(modelname)

else:
    my_model = tensorflow.keras.models.load_model(modelname)

# ----------------- prediction ---------------------
y_pred = my_model.predict(x_test)
y_pred = np.array([np.argmax(i) for i in y_pred])


# ------------------- plot predictions ----------------------------
offset = 250
nrows = 3
ncols = 9
fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
for row in range(0, nrows):
    for col in range(0, ncols):
        axs[row, col].imshow(x_test[offset + nrows * row + col, :, :])
        axs[row, col].set_title('Predicted {}'.format(y_pred[offset + nrows * row + col]))
plt.show()


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=np.arange(0, 10))
disp.plot()
plt.show()


