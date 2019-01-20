import tensorflow as tf 
import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TEST_LEN = 5000

def read(name):
    f = open('./nndata/'+name+'.data','r')
    result = json.loads(f.read())
    f.close()
    return result

    
# In[2]:

#载入数据集
list = np.array(read('list'))
label = np.array(read('label'))

train_list = list[0:(len(list)-TEST_LEN)]
train_label = label[0:(len(label)-TEST_LEN)]

test_list = list[-TEST_LEN:]
test_label = label[-TEST_LEN:]

print('数据加载完毕')
width = 48 #输入一行，一行有28个数据
height = 48 #一共28行
n_classes = 4 # 10个分类
batch_size = 48 #每批次50个样本
n_batch = (len(label)-TEST_LEN)// batch_size #计算一共有多少个批次

x = tf.placeholder(tf.float32, [None, width*height])                        #输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, n_classes])            #输入的标签占位符

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义一个函数，用于构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#构建网络
x_image = tf.reshape(x, [-1,width,height,1])         #转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 5, 1, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层
h_pool1 = max_pool(h_conv1)                                  #第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层
h_pool2 = max_pool(h_conv2)                                   #第二个池化层

W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, n_classes])
b_fc2 = bias_variable([n_classes])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
for j in range(0,10):
    for i in range(0,n_batch):                  
        batch_xs = train_list[(i*batch_size):((i+1)*batch_size)]
        batch_ys = train_label[(i*batch_size):((i+1)*batch_size)]
        #print(batch_xs[-1:])
        if i%50==0:
            train_acc = accuracy.eval(feed_dict={x:batch_xs, y_actual:batch_ys, keep_prob: 1.0})
            print('step ',i,' training accuracy ',train_acc)
        train_step.run(feed_dict={x: batch_xs, y_actual: batch_ys, keep_prob: 0.5})

    test_acc=accuracy.eval(feed_dict={x:test_list,y_actual:test_label, keep_prob: 1.0})
    print('===================================')
    print("test accuracy ",test_acc)
    print('===================================')













