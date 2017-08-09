import tensorflow as tf 
#w = tf.Variable(tf.random_normal([2,1] , stddev=1 , seed = 1 )

#y = tf.matmul(x ,w )

#loss = tf.reduce_mean( tf.square(y_ , y))  + tf.contrib.layers.l2_regularizer(lambd)(w)

weight = tf.constant([[1.0 , -2.0 ],[-3.0 , 4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weight))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weight))
    
