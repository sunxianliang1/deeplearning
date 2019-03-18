import MinstData
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if __name__=='__main__':
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    #y = tf.matmul(x, W) + b
    y_ = tf.placeholder("float", [None,10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))#交叉熵
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer().run()

    mnist=MinstData.MinstData("E:\深度学习\训练集数据\手写字符\压缩版")
    for _ in range(1000):
        batch_xs, batch_ys = mnist.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    right=sess.run(accuracy, feed_dict={x: mnist.TestImg, y_: mnist.TestLabel10})
    #right=accuracy.eval(feed_dict={x: mnist.TestImg, y_: mnist.TestLabel10})
    print(right)

    #for i in range(0, mnist.TestLen):
    #    result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.TestImg[i]]), y_: np.array([mnist.TestLabel10[i]])})
    #    if not result:
    #        print('预测的值是：',sess.run(y, feed_dict={x: np.array([mnist.TestImg[i]]), y_: np.array([mnist.TestLabel10[i]])}))
    #        print('实际的值是：',sess.run(y_,feed_dict={x: np.array([mnist.TestImg[i]]), y_: np.array([mnist.TestLabel10[i]])}))
    #        one_pic_arr = np.reshape(mnist.TestImg[i], (28, 28))
    #        pic_matrix = np.matrix(one_pic_arr, dtype="float")
    #        plt.imshow(pic_matrix)
    #        plt.show()
    #        continue

