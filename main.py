#!/usr/bin/python3
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
from datetime import datetime
import os

import keras
from keras.utils import plot_model
# from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.applications import DenseNet121
import keras.backend as keras_backend

import models.all_conv_net as all_conv_net
import models.densenet_model as densenet
import models.wide_renset_model as wide_resenet_model


BASE_RESULTS_DIR = './results'
N_EPOCHS = 300


def preprocess_data_minus_one_to_one(x):
    x = 2 * ((x - x.min()) / (x.max() - x.min())) - 1  # values from -1 to 1
    return x


def preprocess_data_zero_mean_one_std(x):
    mean = np.mean(x, axis=(0, 1, 2))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[3 - 1] = x.shape[3]
    mean = np.reshape(mean, broadcast_shape)
    x -= mean

    std = np.std(x, axis=(0, 1, 2))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[3 - 1] = x.shape[3]
    std = np.reshape(std, broadcast_shape)
    x /= (std + 1e-6)

    return x


def preprocess_cutoff(input_x):
    "Cutoff Preprocessing Improved Regularization of Convolutional Neural Networks with Cutout"

    p = 0.5
    cutoff_size = 8

    h = input_x.shape[0]
    w = input_x.shape[1]

    mask = np.ones((h, w))

    p_1 = np.random.rand()

    if p_1 > p:
        return input_x
    else:
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - cutoff_size // 2, 0, h)
        y2 = np.clip(y + cutoff_size // 2, 0, h)
        x1 = np.clip(x - cutoff_size // 2, 0, w)
        x2 = np.clip(x + cutoff_size // 2, 0, w)
        # print("{},{},{},{}".format(y1, y2, x1, x2))

        mask[y1:y2, x1:x2] = 0.
        mask = np.expand_dims(mask, axis=-1)

        input_x = mask * input_x

        return input_x


def preprocess_crop(input_x):

    pad = 4
    h_pad = pad // 2

    h = input_x.shape[0]
    w = input_x.shape[1]

    x_enlarged = np.pad(input_x, ((h_pad, h_pad), (h_pad, h_pad), (0, 0)), 'constant')

    start_x = np.random.randint(0, pad)
    start_y = np.random.randint(0, pad)

    x_cropped = x_enlarged[start_y:start_y + w, start_x:start_x + h, :]

    return x_cropped


def preprocess_crop_and_cutout(input_x):
    x = preprocess_crop(input_x)
    x = preprocess_cutoff(x)

    return x


def learning_rate_modifier(epoch_idx, curr_learning_rate):
    if epoch_idx == (N_EPOCHS // 2.0):
        curr_learning_rate = curr_learning_rate / 10.0
    elif epoch_idx == (N_EPOCHS // 4.0 * 3):
        curr_learning_rate = curr_learning_rate / 10.0

    return curr_learning_rate


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    RANDOM_SEED = 100

    n_batch = 64

    # Validation split
    validation_data_split = 0.05

    results_identifier = 'all_convnet_cutout'

    # Immutable ------------------------------------------------
    np.random.seed(RANDOM_SEED)
    plt.ion()

    if not os.path.exists(BASE_RESULTS_DIR):
        os.mkdir(BASE_RESULTS_DIR)

    results_dir = os.path.join(BASE_RESULTS_DIR, results_identifier)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # -----------------------------------------------------------------------------------
    # Load the Data
    # -----------------------------------------------------------------------------------
    print("Loading Data ...")
    # From Kaggle
    with open('./data/train_data', 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
        train_label = pickle.load(f, encoding='bytes')
    with open('./data/test_data', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    # Put data in correct format
    train_data = train_data.reshape((train_data.shape[0], 3, 32, 32))
    train_data = train_data.transpose(0, 2, 3, 1)

    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32))
    test_data = test_data.transpose(0, 2, 3, 1)

    # Built in Keras
    # (train_data, train_label), (X_test, y_test) = cifar100.load_data()

    # split into 67% for train and 33% for test
    X_train, X_validation, y_train, y_validation = \
        train_test_split(train_data, train_label, test_size=validation_data_split, random_state=RANDOM_SEED)
    X_test = test_data

    # Change Labels to one-hot representation
    n_classes = len(set(train_label))
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_validation = keras.utils.to_categorical(y_validation, n_classes)

    # Print Data Set Details
    print("Train Set {}".format(X_train.shape))
    print("Validation Set {}".format(X_validation.shape))
    print("Test Set {}".format(X_test.shape))
    print("Number of classes {}".format(n_classes))
    print("y_train.shape {}".format(y_train.shape))

    # -----------------------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------------------
    print("Building the model ...")
    model = all_conv_net.get_model(n_classes)
    # model = DenseNet121(classes=n_classes, weights=None, input_shape=(32,32,3))
    # keras_backend.set_image_data_format('channels_last')
    # model = densenet.get_model()
    # model = densenet.get_densenet_bc_100_model()
    # model = wide_resenet_model.get_model()

    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.Adam()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.summary()
    plot_model(model, os.path.join(results_dir, 'model_arch.eps'), show_shapes=True)

    # -----------------------------------------------------------------------------------
    # Preprocess the Data
    # -----------------------------------------------------------------------------------
    # TODO: Properly normalize, subtract mean, range (-1, 1)
    X_train = X_train.astype('float32')
    X_validation = X_validation.astype('float32')
    X_test = X_test.astype('float32')

    # X_train /= 255.0
    # X_validation /= 255.0
    # X_test /= 255.0

    X_train = preprocess_data_minus_one_to_one(X_train)
    X_validation = preprocess_data_minus_one_to_one(X_validation)
    X_test = preprocess_data_minus_one_to_one(X_test)

    X_train = preprocess_data_zero_mean_one_std(X_train)
    X_validation = preprocess_data_zero_mean_one_std(X_validation)
    X_test = preprocess_data_zero_mean_one_std(X_test)

    # -----------------------------------------------------------------------------------
    # Train the models
    # -----------------------------------------------------------------------------------
    print("Training the models...")
    start_time = datetime.now()

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=5./32,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=5./32,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,
        preprocessing_function=preprocess_crop_and_cutout
    )

    datagen.fit(X_train, augment=True)
    TEMP_WEIGHT_STORE_FILE = os.path.join(results_dir, "weights.h5")

    checkpoint = ModelCheckpoint(
        TEMP_WEIGHT_STORE_FILE,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
    )

    tensorboard = TensorBoard(
        log_dir='tensorboard_logs/{}'.format(time()),
        # histogram_freq=1,
        # write_grads=True,
        # write_images=False,
        # batch_size=1,  # For histogram
    )

    learning_rate_modifying_cb = LearningRateScheduler(
        learning_rate_modifier,
        verbose=1
    )

    callbacks_list = [checkpoint, tensorboard, learning_rate_modifying_cb]

    # Fit the models on the batches generated by datagen.flow().
    history_callback = model.fit_generator(
        generator=datagen.flow(X_train, y_train, batch_size=n_batch),
        steps_per_epoch=X_train.shape[0] / n_batch,
        epochs=N_EPOCHS,
        validation_data=(X_validation, y_validation),
        callbacks=callbacks_list,
        verbose=1,
    )

    print("Training took {}".format(datetime.now() - start_time))

    # PLot the Training Details
    f, ax_arr = plt.subplots(1, 2)

    history = history_callback.history
    # Save the history of the model
    with open(os.path.join(results_dir, 'history.pkl'), 'wb+') as handle:
        pickle.dump(history, handle)

    ax_arr[0].plot(history['loss'], color='r', label='train')
    ax_arr[0].plot(history['val_loss'], color='b', label='validation')
    ax_arr[0].set_title("Loss")

    ax_arr[1].plot(history['acc'], color='r', label='train')
    ax_arr[1].plot(history['val_acc'], color='b', label='validation')
    ax_arr[1].set_title("Accuracy")
    ax_arr[1].legend()
    f.savefig(os.path.join(results_dir, 'training.eps'), format='eps')

    # -----------------------------------------------------------------------------------
    # Evaluating Test Data
    # -----------------------------------------------------------------------------------
    print("Evaluating the Test Data ")

    y_hat = model.predict(X_test)
    y_hat_max = np.argmax(y_hat, axis=1)

    test_predictions_file = os.path.join(results_dir, 'predictions.csv')
    with open(test_predictions_file, 'w+') as handle:
        handle.write("ids,labels\n")
        for img_idx in np.arange(X_test.shape[0]):
            handle.write(str(img_idx) + ',' + str(y_hat_max[img_idx]) + '\n')
