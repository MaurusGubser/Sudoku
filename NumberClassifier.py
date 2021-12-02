import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class NumberClassifier():

    def __init__(self, input_shape, nb_filters, kernel_size, pool_size):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding_size = pool_size

        self.conv1 = Conv2D(filters=2 * nb_filters, kernel_size=kernel_size, padding='valid', activation=relu)
        self.pool1 = MaxPool2D(pool_size=pool_size)
        self.conv2 = Conv2D(filters=4 * nb_filters, kernel_size=kernel_size, activation=relu)
        self.pool2 = MaxPool2D(pool_size=pool_size)
        self.conv3 = Conv2D(filters=4*nb_filters, kernel_size=kernel_size, activation=relu)
        self.pool3 = MaxPool2D(pool_size=pool_size)
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation=relu)
        self.dense2 = Dense(units=10, activation=softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


nb_filters = 8
kernel_size = 3
pool_size = 2
input_shape = (28, 28, 1)

my_model = Sequential()
my_model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, activation=relu))
my_model.add(Conv2D(2 * nb_filters, kernel_size, activation=relu))
my_model.add(MaxPool2D(pool_size=pool_size))
my_model.add(Conv2D(4 * nb_filters, kernel_size, activation=relu))
my_model.add(MaxPool2D(pool_size=pool_size))
# my_model.add(Conv2D(8 * nb_filters, kernel_size, activation=relu))
# my_model.add(MaxPool2D(pool_size=pool_size))
my_model.add(Flatten())
my_model.add(Dense(64, activation=relu))
my_model.add(Dropout(0.5))
my_model.add(Dense(10, activation=softmax))

my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(my_model.summary())

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

batch_size = 1000
nb_epochs = 10
my_model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs)
# my_model = tf.keras.models.load_model('NumberClassifier')

y_pred = my_model.predict(x=x_test)
y_pred = np.array([np.argmax(i) for i in y_pred])

"""
fig, axs = plt.subplots(nrows=3, ncols=5)
for row in range(0, 3):
    for col in range(0, 5):
        axs[row, col].imshow(x_test[3 * row + col, :, :, :])
        axs[row, col].set_title('Predicted {}'.format(y_pred[3 * row + col]))
plt.show()
"""

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=np.arange(0, 10))
disp.plot()
plt.show()

modelname = 'NumberClassifierNormal'
my_model.save(modelname)
