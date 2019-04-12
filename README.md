# MNIST 手写识别


## 全连接实现：
  首先导入依赖项:
  ```
  import tensorflow as tf
  import pylab
  from tensorflow.examples.tutorials.mnist import input_data
  ```
读取mnist:
```
mnist = input_data.read_data_sets("/data/", one_hot=True)
```
参数设置:
```
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1
```

设置神经网络节点数:
```
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 =512  # 2nd layer number of features
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)
```
搭建计算图:
```
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#定义各层权重与偏置值
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#定义两层神经网络
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
```
优化
```
#构建模型
pred = multilayer_perceptron(x, weights, biases)
#定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#梯度下降最小化损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
```
计算图搭建好了，启动会话
```
with tf.Session() as sess:
    sess.run(init)

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print (" Finished!")
```
测试 model
```
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval,predv = sess.run([output,pred], feed_dict={x: batch_xs})
    print(outputval,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
```
效果图：
<br>
![image](https://github.com/qianlongql/a-world/blob/master/g.PNG)
 
 
 正确率为0.9622
 输出[1,6]为预测值
 还是非常准确的哈^_^
