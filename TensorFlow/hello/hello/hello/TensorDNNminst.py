import tensorflow as tf
import MinstData
import matplotlib.pyplot as plt
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
权重初始化
初始化为一个接近0的很小的正数
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

if __name__=='__main__':
    x = tf.placeholder("float", [None, 784])
    W1 = weight_variable([784,90])
    b1 = bias_variable([90])
    x1=tf.nn.relu_layer(x, W1,b1)
    #x11 = tf.matmul(x, W1) + b1
    #x1=tf.nn.sigmoid(x11)
    W2 = weight_variable([90,10])
    b2 = bias_variable([10])
    #x2 = tf.matmul(x1, W2) + b2
    x2=tf.nn.relu_layer(x1, W2,b2)
    y=tf.nn.softmax(x2)
    y_ = tf.placeholder("float", [None,10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))#交叉熵
    train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mnist=MinstData.MinstData("E:\deeplearning\训练集数据\手写字符\压缩版")
    for _ in range(1000):
        batch_xs, batch_ys = mnist.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    right=sess.run(accuracy, feed_dict={x: mnist.TestImg, y_: mnist.TestLabel10})
    print(right)
























