# MNIST 手写识别

## 卷积神经网络实现：

### 导入依赖项：
```
import tensorflow as tf
import pylab
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)
```
### 定义卷积核、偏置值：
```
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```
### 定义卷积层，步长为1，边框补0
```
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```
### 定义池化层为最大池化，窗口大小为2x2，步长为2：
```
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```
### 定义平均池化层，窗口大小为7x7，步长为7:
```
def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')
```
* 这里对图片、卷积核、池化层窗口和步长的shape做一个备注：
  * 图像的 shape=[ 每一批的图像数（batch) ， 图像高度 ，图像宽度 ，图像通道数 ]
  * 卷积核(filter)的 shape=[ 核高度，核宽度 ，图像通道数 ，核的个数 ], 有多少个核就生成多少个feature map
  * 池化窗口(ksize)的 shape 一般为[ 1 , 高 ，宽 ，1 ]
  * 步长一般为 [ 1, 长方向上的步长 ， 宽方向上的步长 ，1]

### 搭建模型：
```
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

#定义卷积核，由于张量不太好理解，本人试着画了一下感觉shape不太对，不过不要紧，按照上面的备注来理解
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])#定义偏置值

x_image = tf.reshape(x, [-1,28,28,1])#卷积需要将数据变成长×宽的shape

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#第一层卷积，生成32张28x28的feature map
h_pool1 = max_pool_2x2(h_conv1)#第一层池化，生成32张14x14的feature map

W_conv2 = weight_variable([5, 5, 32, 64])#定义64个5x5的卷积核，图像通道为32
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#第二层卷积，生成64张14x14的feature map
h_pool2 = max_pool_2x2(h_conv2)#第二层池化，生成64张7x7的feature map

W_conv3 = weight_variable([5, 5, 64, 10])#5x5的10个卷积核
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)#第三层卷积，生成10张7x7的feature map
nt_hpool3=avg_pool_7x7(h_conv3)##第三层池化，生成10张1x1的feature

nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])#reshape成10维向量，代表10分类的各个分类的比重
y_conv=tf.nn.softmax(nt_hpool3_flat)#softmax归一化，让每一维的数字代表对应分类的概率
```

### 优化：
```
loss = -tf.reduce_sum(y*tf.log(y_conv))#计算损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)#梯度下降最小化loss

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
#argmax取输入向量的最大值的索引，equal返回两者相等的真值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#强制转换为float,计算平均值即为正确率
```
### 启动session:
```
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
```

