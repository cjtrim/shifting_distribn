from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm


# global experiment vars
# Directory storing the saved model.
train_dir = "/tmp"
# Filename to save model under.
def filename(iteration):
    return "cifar2_"+str(iteration)+".ckpt"
# Number of epochs to train model
nb_epochs = 30
# Size of training batches
batch_size = 32
# Learning rate for training
learning_rate = 0.001
# Max number of algo iterations.
maxIter = 20
# Epsilon for FGSM
epsilon = 0.2
# Number of classes
num_classes = 2

def main(args):

    # Make sure tf is backend for keras.
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Import data.
    X_train, Y_train, X_test, Y_test = data_cifar(num_classes)

    # Run algorithm to convergence or for maximum iterations.
    converged = False
    nets = []
    for i in range(maxIter):
        if converged:
            break

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 2))

        # Define TF model graph
        model = cnn_model(X_train.shape[1:], Y_train.shape[1])
        predictions = model(x)
        print("Defined TensorFlow model graph.")

        # build net on current data set
        model = cnn_model()

        ## train net
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                      metrics=['accuracy'])

        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epochs,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

        # generate adversarial examples
        adv_x = fgsm(x, predictions, eps=epsilon)
        eval_params = {'batch_size': batch_size}
        X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
        assert X_test_adv.shape[0] == 2000

        # add missed examples to the datasets

        # test for 'convergence'
        disp_imgs = []
        for i in range(10):
            # display test images and wait for input
            X_test_adv = X_test_adv.astype('float32')
            img_adv = to_image(X_test_adv[1])
            img_adv.show()
            disp_imgs.append(img_adv)

        converged = 'Y' in raw_input("Are these images recognizable? (y/n) ").upper()

        for img_adv in disp_imgs:
            img_adv.close()

        print("Saving model as "+train_dir+'/'+filename(i))
        model.save(train_dir+'/'+filename(i))
        # retrain
        # repeat until images are unrecogizable
    # test resultant randomized classifier based on prob


    pass


def data_cifar(nb_classes):
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = max(num_classes, 10)

    # the data, shuffled and split between train and test sets
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    inds_train = [i for i, e in enumerate(Y_train) if e in range(nb_classes)]
    inds_test = [i for i, e in enumerate(Y_test) if e in  range(nb_classes)]

    X_train = X_train[inds_train]
    Y_train = Y_train[inds_train]
    X_test = X_test[inds_test]
    Y_test = Y_test[inds_test]

    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def cnn_model(in_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


    if __name__ == '__main__':
        app.run()
