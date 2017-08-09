#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:54:18 2017

@author: root
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:55:54 2017

@author: root
"""
#三个步骤
#定义神经网络的结构和前向传播的输出结果
#定义损失函数以及选择反向传播优化的算法
#生成会话，并且在训练数据上反复运行反向传播优化算法

import tensorflow as tf
import numpy as np


batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3], stddev=1  , seed = 1 ))
w2 = tf.Variable(tf.random_normal([3,1] , stddev=1 , seed = 1 ))
#
x = tf.placeholder(tf.float32 , shape=[None , 2] , name="x-input")
y_ = tf.placeholder(tf.float32 , shape=[None , 1] , name="y-input")

a = tf.matmul(x , w1)
y = tf.matmul(a, w2)

#cross_entropy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value( y , 1e-10 , 1.0)))
#clip_by_value 大致的意思是：限定y的大小，不能小于1-e10，不能大于1.0 


loss_less = 10 
loss_more = 1 

loss  = tf.reduce_sum( tf.select(tf.greater(y,y_) , (y - y_)*loss_more , (y_- y)*loss_less))

#loss = tf.reduce_mean( tf.select())
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
train_step = tf.AdamOptimizer(0.001).minimize(loss)


rdm = np.random.RandomState(1)

dataset_size = 128
X = rdm.rand(dataset_size  , 2 )

Y = [[int(x1+x2 < 1 )] for (x1 , x2 ) in X ]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    #print sess.run(w1)
    #print sess.run(w2)
    
    
    STEPS = 5000
    
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size , dataset_size)
        #print "batch process : " , start , end 
        sess.run( train_step , feed_dict= {x : X[start :end] , y_ : Y[start:end]})
        if i %1000 == 0 :
            total_loss = sess.run( loss , feed_dict = {x : X , y_ : Y})
            print("After %d training step(s) , cross entropy on all data is %g" % (i , total_loss))
            
    print sess.run(w1)
    print sess.run(w2)
