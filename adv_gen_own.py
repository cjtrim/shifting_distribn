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
#from cleverhans.utils import cnn_model

from PIL import Image
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'cifar2.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 32, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')



def main(argv=None):
    """
    CIFAR10 cleverhans tutorial
    :return:
    """
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

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

    ### GET DATA

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 2

    # the data, shuffled and split between train and test sets
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # keep only the important classes
    inds_train = [i for i, e in enumerate(Y_train) if e in range(nb_classes)]
    inds_test = [i for i, e in enumerate(Y_test) if e in range(nb_classes)]
    X_train = X_train[inds_train]
    Y_train = Y_train[inds_train]
    X_test = X_test[inds_test]
    Y_test = Y_test[inds_test]



    # preprocess the data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # # smoothing?
    # assert Y_train.shape[1] == 2.
    # label_smooth = .1
    # Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    # Define TF model graph
    model = cnn_model(X_train.shape[1:], Y_train.shape[1])
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    ### TRAIN

    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    model_train(sess, x, y, predictions, X_train, Y_train,
                evaluate=evaluate, args=train_params)

    """
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    """
    img = to_image(X_train[1])
    img.show()

    model.fit(X_train, Y_train,
              batch_size=FLAGS.batch_size,
              nb_epoch=FLAGS.nb_epochs,
              validation_data=(X_test, Y_test),
              shuffle=True)

    print("Done")
    quit()
    ### GENERATE ADV IMGS

    ### RETRAIN

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    eval_params = {'batch_size': FLAGS.batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
    assert X_test_adv.shape[0] == 2000

    X_test_adv = X_test.astype('float32')
    X_test_adv /= 255

    img_adv = to_image(X_test_adv[1])
    img_adv.show()

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = cnn_model(X_train.shape[1:], Y_test.shape[1])
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained CIFAR10 model on
        # legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained CIFAR10 model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test,
                                  Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    model_train(sess, x, y, predictions_2, X_train, Y_train,
                predictions_adv=predictions_2_adv, evaluate=evaluate_2,
                args=train_params)

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions_2_adv, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

def to_image(ex):
    example = ex * 255
    example = example.astype('uint8')
    if len(example.shape) == 4:
        img = [Image.fromarray(i, 'RGB') for i in example]
    else:
        img = Image.fromarray(example, 'RGB')
    return img

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
