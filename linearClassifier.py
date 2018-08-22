
# Reference: http://learningtensorflow.com/classifying/

from sklearn.datasets import make_blobs

import numpy as np

from sklearn.preprocessing import OneHotEncoder

# Three blobs of data containing 800 samples, each sample has the feature consist of 3 elements
X_values, y_flat = make_blobs(n_features=3, n_samples=800, centers=3, random_state=500)

# Apply One Hot Encoder, [2] -> [0 0 1]
y = OneHotEncoder().fit_transform(y_flat.reshape(-1, 1)).todense()
y = np.array(y)


from sklearn.model_selection import train_test_split

# Split dataset into training and testing set
X_train, X_test, y_train, y_test, y_train_flat, y_test_flat = train_test_split(X_values, y, y_flat)

# Add noise on test data
X_test += np.random.randn(*X_test.shape) * 1.5

import tensorflow as tf

n_features = X_values.shape[1]
n_classes = len(set(y_flat))

weights_shape = (n_features, n_classes)

W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model

X = tf.placeholder(dtype=tf.float32)

Y_true = tf.placeholder(dtype=tf.float32)

bias_shape = (1, n_classes)
b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))

Y_pred = tf.matmul(X, W)  + b

# Define loss function and learning rate
loss_function = tf.losses.softmax_cross_entropy(Y_true, Y_pred)
learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(5000):

        result = sess.run(learner, {X: X_train, Y_true: y_train})

        if i % 100 == 0:
            print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(loss_function, {X: X_test, Y_true: y_test})))

    # After training, do prediction on X_test
    y_pred = sess.run(Y_pred, {X: X_test})
    W_final, b_final = sess.run([W, b])

# Find the indices of maximal likelihood of each row, the indices represent the class ID
predicted_y_values = np.argmax(y_pred, axis=1)

# Calculate Accuracy
hit = 0

compare = predicted_y_values - y_test_flat

nonzero_index = np.nonzero(compare)

miss_cnt = np.size(nonzero_index, axis = 1)

accuracy = 1 - miss_cnt/len(y_test_flat)

print('\nAccuracy: {}%'.format(accuracy * 100))
