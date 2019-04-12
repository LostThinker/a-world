# -*- coding: utf-8 -*-
import tensorflow as tf
import pylab
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

#设置模型保存路径
saver = tf.train.Saver()
model_path = "log/mnistmodel.ckpt"

#定义卷积核、偏置值
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义卷积层，步长为1，边框补0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#定义池化层为最大池化，窗口大小为2x2，步长为2
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#定义平均池化层，窗口大小为7x7，步长为7
def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes



W_conv1 = weight_variable([5, 5, 1, 32])#定义卷积核，但是不太能理解shape，自己画出来的张量不是32个5x5的矩阵
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])#卷积需要将数据变成长×宽的shape

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#第一层卷积，生成32张28x28的feature map
h_pool1 = max_pool_2x2(h_conv1)#第一层池化，生成32张14x14的feature map

W_conv2 = weight_variable([5, 5, 32, 64])#定义64个5x5的卷积核，图像通道为32
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#第二层卷积，生成64张14x14的feature map
h_pool2 = max_pool_2x2(h_conv2)#第二层池化，生成64张7x7的feature map
#########################################################new
W_conv3 = weight_variable([5, 5, 64, 10])#5x5的10个卷积核
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)#第三层卷积，生成10张7x7的feature map

nt_hpool3=avg_pool_7x7(h_conv3)##第三层池化，生成10张1x1的feature
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])#reshape成10维向量，代表10分类的各个分类的概率
y_conv=tf.nn.softmax(nt_hpool3_flat)#softmax归一化，让每一维的数字代表对应分类的概率


loss = -tf.reduce_sum(y*tf.log(y_conv))#计算损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)#梯度下降最小化loss

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
#argmax取输入向量的最大值的索引，equal返回两者相等的真值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#强制转换为float,计算平均值即为正确率

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):#20000
      batch = mnist.train.next_batch(50)#50
      if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1]})
        print( "step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y: batch[1]})
    
    print ("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels}))
    
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    print('-------------------------------')

    output = tf.argmax(y_conv, 1)
    batch_xs, batch_ys = mnist.train.next_batch(1)
    batch_xx,batch_yy=mnist.test.next_batch(10)
    
    outputval,predv = sess.run([output,y_conv], feed_dict={x: batch_xs})
    outputval2,predv2 = sess.run([output,y_conv], feed_dict={x: batch_xx})
    
    print('---------------------------')
    print("训练集测试")
    print("预测结果",outputval,'\n',batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    print('---------------------------')
    print("测试集测试")
    print("预测结果: ",outputval2,'\n',batch_yy)
    for i in range(10):
        im = batch_xx[i]
        im = im.reshape(-1,28)
        pylab.imshow(im)
        pylab.show()

'''
print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    
     # 测试 model
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    output = tf.argmax(y_conv, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval,predv = sess.run([output,y_conv], feed_dict={x: batch_xs})
    print(outputval,predv,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
'''
    
    
    