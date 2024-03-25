import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.models
import tf2onnx
import onnxruntime as ort
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2 as cv


def define_neural_network(nb_filters, input_shape, kernel_size, pool_size, dense_layer_size):
    """
    Define layers of neural network
    :param nb_filters: int
        number of filters to use in convolutional layers, doubled in each step
    :param input_shape: tuple of int
        integers, which describe the shape of the input
    :param kernel_size: int or (int, int)
        integer or tuple of integers, which describe the shape of the filter
    :param pool_size: int
        integer, which describe the pool_size for convolutional layer
    :param dense_layer_size: int
        integer, which sets the size of the dense layer
    :return: Sequential
        a sequential keras CNN model
    """
    my_model = Sequential()
    my_model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, activation=relu))
    my_model.add(MaxPool2D(pool_size=pool_size))
    my_model.add(Conv2D(2 * nb_filters, kernel_size, activation=relu))
    my_model.add(MaxPool2D(pool_size=pool_size))
    my_model.add(Flatten())
    my_model.add(Dense(dense_layer_size, activation=relu))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(10, activation=softmax))
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(my_model.summary())
    return my_model


def get_number_from_path(path):
    start = path.rfind('/')
    if start >= 0:
        number = int(path[start + 1:])
        return number
    else:
        start = path.rfind('\\')
        number = int(path[start + 1:])
        return number


def load_european_digits_data(path):
    x = []
    y = []
    for root, dirs, files in os.walk(path):
        if not dirs:
            nb = get_number_from_path(root)
            for file in files:
                path_img = root + '/' + file
                img = 255 - cv.imread(path_img, cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (28, 28))
                x.append(img)
                y.append(nb)
    return np.array(x), np.array(y)


def load_train_test_data(nb_mnist_digits, use_european_digits, use_only_ones):
    (x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = tf.keras.datasets.mnist.load_data()
    x = np.append(x_mnist_train, x_mnist_test, axis=0)[0:nb_mnist_digits]
    y = np.append(y_mnist_train, y_mnist_test, axis=0)[0:nb_mnist_digits]
    if use_european_digits:
        path_european_data = 'EuropeanData'
        x_european, y_european = load_european_digits_data(path_european_data)
        if use_only_ones:
            y_ones = y_european[y_european == 1]
            x_ones = x_european[y_european == 1]
            x = np.append(x, x_ones, axis=0)
            y = np.append(y, y_ones, axis=0)
        else:
            x = np.append(x, x_european, axis=0)
            y = np.append(y, y_european, axis=0)

    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print('Train size={}, Test size={}'.format(y_train.size, y_test.size))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    return x_train, y_train, x_test, y_test


# ----------------- data ---------------------
nb_mnist = 50000
use_european_digits = False
use_only_ones = False
x_train, y_train, x_test, y_test = load_train_test_data(nb_mnist, use_european_digits, use_only_ones)

# ----------------- model ---------------------
train = False
modelname = 'NumberClassifier_MNIST'
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
    nb_epochs = 10
    my_model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs)
    my_model.save(modelname + '.h5')
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input'),)
    model_proto, _ = tf2onnx.convert.from_keras(my_model,
                                                input_signature=spec,
                                                opset=13,
                                                output_path=modelname + '.onnx')
else:
    my_model = tensorflow.keras.models.load_model(modelname+'.h5')
    # my_model = onnx.load(modelname + '.onnx')

# ----------------- prediction ---------------------
y_pred = my_model.predict(x_test)
y_pred = np.array([np.argmax(i) for i in y_pred])

ort_sess = ort.InferenceSession(modelname + '.onnx')
y_pred = ort_sess.run(None, {'data': x_test})

# ------------------- plot predictions ----------------------------
offset = 250
nrows = 3
ncols = 10
fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
for row in range(0, nrows):
    for col in range(0, ncols):
        axs[row, col].imshow(x_test[offset + ncols * row + col, :, :])
        axs[row, col].set_title('Predicted {}'.format(y_pred[offset + ncols * row + col]))
plt.show()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=np.arange(0, 10))
disp.plot()
plt.show()
