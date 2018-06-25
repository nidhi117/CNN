import math
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import pydicom
from tensorflow.python.framework import ops

# To get pixels from dicom images
def extract_pixels_from_dicom(filename):
    ds = pydicom.read_file(filename, stop_before_pixels=False, force=True)
    return ds.pixel_array

# Function to create batche
def batch(df, bstrt, bend):
    X = list()
    Y = list()
    c = 0
    for x in range(bstrt, bend):
        pixels = extract_pixels_from_dicom(str(df.iloc[x, 0]))
        if len(pixels[0]) == 128:
            y = df.iloc[x, 1]
            X.append(pixels)
            Y.append(y)
    # print(len(X[0]))
    # print(len(X[0][0]))
    # print("Batch Created!")
    return X, Y


def create_placeholders(n_H0=128, n_W0=128, n_C0=1, n_y=3):
    X = tf.placeholder(tf.float32, [None, 128, 128, 1])
    Y = tf.placeholder(tf.int64, [None, 3])
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}
    return parameters

# Network architecture
def forward_propagation(X, parameters):
    print("Forward Prop!")
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(tf.contrib.layers.batch_norm(Z1))
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(tf.contrib.layers.batch_norm(Z2))
    # MAXPOOL: window 2x2, stride 2, padding 'VALID'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # CONV2D: filters W3 stride of 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A3 = tf.nn.relu(tf.contrib.layers.batch_norm(Z3))
    # MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='VALID')  
    # CONV2D: filters W4, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A4 = tf.nn.relu(tf.contrib.layers.batch_norm(Z4))
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
    print("P4_flattened: ", P4.shape)
    # FULLY-CONNECTED layers
    Z5 = tf.contrib.layers.fully_connected(tf.contrib.layers.batch_norm(P4), 4096, activation_fn=tf.nn.relu)
    Z6 = tf.contrib.layers.fully_connected(tf.contrib.layers.batch_norm(Z5), 512, activation_fn=tf.nn.relu)
    Z7 = tf.contrib.layers.fully_connected(tf.contrib.layers.batch_norm(Z6), 64, activation_fn=tf.nn.relu)
    Z8 = tf.contrib.layers.fully_connected(tf.contrib.layers.batch_norm(Z7), 3, activation_fn=None)
    return Z8


def compute_cost(Z8, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z8, labels=Y))
    return cost


def model(learning_rate=0.009, num_epochs=50):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    df_img_path = pd.read_csv('./new_ds_shuffled.csv', index_col=False)
    df_train_compete = df_img_path.loc[0:34855]
    df_train1 = df_img_path.loc[7168:34855]
    df_valid1 = df_img_path.loc[0:7167]
    df_valid2 = df_img_path.loc[7168:14335]
    df_train2 = df_img_path.loc[0:7167]
    df_train2 = df_train2.append(df_img_path.loc[14336:34855])
    df_valid3 = df_img_path.loc[14336:21503]
    df_train3 = df_img_path.loc[0:14335]
    df_train3 = df_train3.append(df_img_path.loc[21504:34855])
    df_valid4 = df_img_path.loc[21504:28671]
    df_train4 = df_img_path.loc[0:21503]
    df_train4 = df_train4.append(df_img_path.loc[28672:34855])
    df_valid5 = df_img_path.loc[28672:34855]
    df_train5 = df_img_path.loc[0:28672]
    df_test = df_img_path.loc[34856:]
    tf.set_random_seed(1)
    seed = 3
    costs = []
    X, Y = create_placeholders()
    parameters = initialize_parameters()
    Z8 = forward_propagation(X, parameters)
    cost = compute_cost(Z8, Y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    num_batches_train = 54
    num_batches_test = 8
    valid = 5
    batch_size = 512
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        valid_acc = 0
        predict_op = tf.argmax(Z8, 1)
        correct_prediction = tf.equal(predict_op,tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            bend = 0
            for i in range(num_batches_train):
                data, y = batch(df_train1, i * batch_size, (i + 1) * batch_size)
                data = np.array(data)
                # print(data.ndim)
                y = np.array(y)
                b = np.zeros((len(data), 3))
                b[np.arange(len(data)), y] = 1
                data = np.reshape(data, (len(data), 128, 128, 1))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: data, Y: b})
                print('Batch no : %i, Epoch no : %i' % (i + 1, epoch + 1))
                mcost += temp_cost / num_batches_train

            # Print the cost every epoch
            print("Cost after epoch %i: %f" % (epoch + 1, mcost))
            costs.append(mcost)
        po1 = tf.argmax(Z8, 1)
        cp1 = tf.equal(po1,tf.argmax(Y,1))
        acc1 = tf.reduce_mean(tf.cast(cp1, "float"))
        for k in range(1):
            data, y = batch(df_valid1, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            valid_acc += acc1.eval({X: data, Y: b})
        print("Validation Accuracy after 1st validation:" , valid_acc)
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            bend = 0
            for i in range(num_batches_train):
                data, y = batch(df_train2, i * batch_size, (i + 1) * batch_size)
                data = np.array(data)
                y = np.array(y)
                b = np.zeros((len(data), 3))
                b[np.arange(len(data)), y] = 1
                data = np.reshape(data, (len(data), 128, 128, 1))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: data, Y: b})
                print('Batch no : %i, Epoch no : %i' % (i + 1, epoch + 1))
                mcost += temp_cost / num_batches_train

            # Print the cost every epoch
            print("Cost after epoch %i: %f" % (epoch + 1, mcost))
            costs.append(mcost)
        po2 = tf.argmax(Z8, 1)
        cp2 = tf.equal(po2,tf.argmax(Y,1))
        acc2 = tf.reduce_mean(tf.cast(cp2, "float"))
        for k in range(1):
            data, y = batch(df_valid2, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            valid_acc += acc2.eval({X: data, Y: b})
        print("Validation Accuracy after 2nd validation:" , valid_acc/2)
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            bend = 0
            for i in range(num_batches_train):
                data, y = batch(df_train3, i * batch_size, (i + 1) * batch_size)
                data = np.array(data)
                y = np.array(y)
                b = np.zeros((len(data), 3))
                b[np.arange(len(data)), y] = 1
                data = np.reshape(data, (len(data), 128, 128, 1))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: data, Y: b})
                print('Batch no : %i, Epoch no : %i' % (i + 1, epoch + 1))
                mcost += temp_cost / num_batches_train

            # Print the cost every epoch
            print("Cost after epoch %i: %f" % (epoch + 1, mcost))
            costs.append(mcost)
        po3 = tf.argmax(Z8, 1)
        cp3 = tf.equal(po3,tf.argmax(Y,1))
        acc3 = tf.reduce_mean(tf.cast(cp3, "float"))
        for k in range(1):
            data, y = batch(df_valid3, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            valid_acc += acc3.eval({X: data, Y: b})
        print("Validation Accuracy after 3rd validation:" , valid_acc/3)
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            bend = 0
            for i in range(num_batches_train):
                data, y = batch(df_train4, i * batch_size, (i + 1) * batch_size)
                data = np.array(data)
                # print(data.ndim)
                y = np.array(y)
                b = np.zeros((len(data), 3))
                b[np.arange(len(data)), y] = 1
                data = np.reshape(data, (len(data), 128, 128, 1))
                # print(type(X))
                # print(type(Y))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: data, Y: b})
                print('Batch no : %i, Epoch no : %i' % (i + 1, epoch + 1))
                mcost += temp_cost / num_batches_train

            # Print the cost every epoch
            print("Cost after epoch %i: %f" % (epoch + 1, mcost))
            costs.append(mcost)
        po4 = tf.argmax(Z8, 1)
        cp4 = tf.equal(po4,tf.argmax(Y,1))
        acc4 = tf.reduce_mean(tf.cast(cp4, "float"))
        for k in range(1):
            data, y = batch(df_valid4, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            valid_acc += acc4.eval({X: data, Y: b})
        print("Validation Accuracy after 4th validation:" , valid_acc/4)
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            bend = 0
            for i in range(num_batches_train):
                data, y = batch(df_train5, i * batch_size, (i + 1) * batch_size)
                data = np.array(data)
                y = np.array(y)
                b = np.zeros((len(data), 3))
                b[np.arange(len(data)), y] = 1
                data = np.reshape(data, (len(data), 128, 128, 1))
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: data, Y: b})
                print('Batch no : %i, Epoch no : %i' % (i + 1, epoch + 1))
                mcost += temp_cost / num_batches_train

            # Print the cost every epoch
            print("Cost after epoch %i: %f" % (epoch + 1, mcost))
            costs.append(mcost)
        po5 = tf.argmax(Z8, 1)
        cp5 = tf.equal(po5,tf.argmax(Y,1))
        acc5 = tf.reduce_mean(tf.cast(cp5, "float"))
        for k in range(1):
            data, y = batch(df_valid5, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            valid_acc += acc5.eval({X: data, Y: b})
        print("Validation Accuracy after 5th validation:" , valid_acc/5)

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict = tf.argmax(Z8, 1)
        correct_predict = tf.equal(predict, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        acc = tf.reduce_mean(tf.cast(correct_predict, "float"))
        print(accuracy)
        print("Model Trained!")
        train_accuracy = 0
        test_accuracy = 0
        for k in range(num_batches_train):
            data, y = batch(df_train_compete, k * batch_size, (k + 1) * batch_size)
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            train_accuracy += accuracy.eval({X: data, Y: b})
        print("Training Accuracy: ",train_accuracy/68)
        for k in range(num_batches_test):
            data, y = batch(df_test, k * batch_size, (k + 1) * (batch_size-1))
            data = np.array(data)
            y = np.array(y)
            b = np.zeros((len(data), 3))
            b[np.arange(len(data)), y] = 1
            data = np.reshape(data, (len(data), 128, 128, 1))
            test_accuracy += acc.eval({X: data, Y: b})
        print("Train Accuracy:", train_accuracy / 73)
        print("Validation Accuracy:", valid_acc / 6)
        print("Test Accuracy:", test_accuracy / num_batches_test)
        save_path = saver.save(sess, "./model-main-50.ckpt")

    return train_accuracy, valid_acc, test_accuracy, parameters

_, _, _, parameters = model()
