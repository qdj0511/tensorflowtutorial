# -*- coding:utf-8 -*-

import tensorflow as tf


"""
滑动平均模型

"""
#定义一个变量用于计算滑动平均，这个变量的初始值为0，。注意这里手动指定了变量的类型为tf.float32 ，因为所有需要计算滑动平均的变量必须是实数型。
#
v1 = tf.Variable(0, dtype = tf.float32)

#这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减
step = tf.Variable( 0 , trainable = False)

#定义一个滑动平均的类，初始化时给定衰减因子（0.99） 和控制衰减的变量step

ema = tf.train.ExponentialMovingAverage(0.99 , step)

maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    init_opt = tf.initialize_all_variables()
    sess.run(init_opt)

    #通过ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均都为0
    print sess.run([v1 , ema.average(v1)])


    #变量更新为5
    sess.run(tf.assign(v1, 5))


    sess.run(maintain_average_op)



    print sess.run([v1,ema.average(v1)])

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))

    sess.run(maintain_average_op)

    print sess.run([v1,ema.average(v1)])


    sess.run(maintain_average_op)

    print sess.run([v1,ema.average(v1)])
